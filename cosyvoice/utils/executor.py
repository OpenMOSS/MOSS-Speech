# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import shutil 
import logging
from contextlib import nullcontext
import os
import torchaudio
import torch
import torch.distributed as dist
import torchaudio
from cosyvoice.utils.train_utils import update_parameter_and_lr, log_per_step, log_per_save, batch_forward, batch_backward, save_model, cosyvoice_join
import datetime
import sys
from datetime import timedelta
sys.path.append('/inspire/hdd/project/embodied-multimodality/public/lzjjin/CosyVoice/cosyvoice/utils')
from file_utils import get_dataset_name_from_path
class Executor:

    def __init__(self, gan: bool = False, ref_model: torch.nn.Module = None, dpo_loss: torch.nn.Module = None):
        self.gan = gan
        self.ref_model = ref_model
        self.dpo_loss = dpo_loss
        self.step = 0
        self.epoch = 0
        self.validate_interval=None
        self.rank = int(os.environ.get('RANK', 0))
        self.device = torch.device('cuda:{}'.format(self.rank))

    def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=None):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        if self.ref_model is not None:
            self.ref_model.eval()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict, ref_model=self.ref_model, dpo_loss=self.dpo_loss)
                    info_dict = batch_backward(model, scaler, info_dict)

                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    batch_dict['turn'] = 'discriminator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                with context():
                    batch_dict['turn'] = 'generator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)


    # def train_one_epoc(self, model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, scaler, group_join, ref_model=None):
    #     ''' Train one epoch
    #     '''

    #     lr = optimizer.param_groups[0]['lr']
    #     logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
    #     logging.info('using accumulate grad, new batch size is {} times'
    #                  ' larger than before'.format(info_dict['accum_grad']))
    #     # A context manager to be used in conjunction with an instance of
    #     # torch.nn.parallel.DistributedDataParallel to be able to train
    #     # with uneven inputs across participating processes.
    #     model.train()
    #     if self.ref_model is not None:
    #         self.ref_model.eval()
    #     model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
    #     train_loader_iter = iter(train_data_loader)
    #     info_dict["tag"] = "TRAIN"
    #     info_dict["epoch"] = self.epoch
    #     with model_context():
    #         batch_idx = -1
    #         while True:
    #             batch_idx += 1
    #             data_exhausted_local = False
    #             try:
    #                 current_batch_dict = next(train_loader_iter)
    #             except StopIteration:
    #                 data_exhausted_local = True
    #             data_exhausted_global_signal = torch.tensor([int(data_exhausted_local)], dtype=torch.int, device=self.device)
    #             dist.all_reduce(data_exhausted_global_signal, op=dist.ReduceOp.MAX, group=group_join)
    #             if data_exhausted_global_signal.item() == 1:
    #                 break 
    #             batch_dict = current_batch_dict

    #             torch.cuda.empty_cache()
    #             info_dict["step"] = self.step
    #             info_dict["batch_idx"] = batch_idx
                
    #             # if cosyvoice_join(group_join, info_dict):
    #             #     break

    #             if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
    #                 context = model.no_sync
    #             else:
    #                 context = nullcontext

    #             with context():
    #                 info_dict = batch_forward(model, batch_dict, scaler, info_dict, ref_model=self.ref_model, dpo_loss=self.dpo_loss)
    #                 info_dict = batch_backward(model, scaler, info_dict)

    #             info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
    #             log_per_step(writer, info_dict)
    #             if info_dict.get('save_per_step', 0) > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
    #                (batch_idx + 1) % info_dict["accum_grad"] == 0:
    #                 dist.barrier()
    #                 self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
    #                 model.train()
    #             if (batch_idx + 1) % info_dict["accum_grad"] == 0:
    #                 self.step += 1
    #     dist.barrier()
    #     self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)
        
    def train_one_epoc_gan(self, model, optimizer, scheduler, optimizer_d, scheduler_d, train_data_loader, cv_data_loader,
                           writer, info_dict, scaler, group_join):
        ''' Train one epoch
        '''

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {} rank {}'.format(self.epoch, lr, self.rank))
        logging.info('using accumulate grad, new batch size is {} times'
                     ' larger than before'.format(info_dict['accum_grad']))
        # A context manager to be used in conjunction with an instance of
        # torch.nn.parallel.DistributedDataParallel to be able to train
        # with uneven inputs across participating processes.
        model.train()
        model_context = model.join if info_dict['train_engine'] == 'torch_ddp' else nullcontext
        with model_context():
            for batch_idx, batch_dict in enumerate(train_data_loader):
                import pdb
                pdb.set_trace()
                info_dict["tag"] = "TRAIN"
                info_dict["step"] = self.step
                info_dict["epoch"] = self.epoch
                info_dict["batch_idx"] = batch_idx
                if cosyvoice_join(group_join, info_dict):
                    break

                # Disable gradient synchronizations across DDP processes.
                # Within this context, gradients will be accumulated on module
                # variables, which will later be synchronized.
                if info_dict['train_engine'] == 'torch_ddp' and (batch_idx + 1) % info_dict["accum_grad"] != 0:
                    context = model.no_sync
                # Used for single gpu training and DDP gradient synchronization
                # processes.
                else:
                    context = nullcontext

                with context():
                    batch_dict['turn'] = 'discriminator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer_d, scheduler_d, scaler, info_dict)
                optimizer.zero_grad()
                log_per_step(writer, info_dict)
                with context():
                    batch_dict['turn'] = 'generator'
                    info_dict = batch_forward(model, batch_dict, scaler, info_dict)
                    info_dict = batch_backward(model, scaler, info_dict)
                info_dict = update_parameter_and_lr(model, optimizer, scheduler, scaler, info_dict)
                optimizer_d.zero_grad()
                log_per_step(writer, info_dict)
                # NOTE specify save_per_step in cosyvoice.yaml if you want to enable step save
                if info_dict['save_per_step'] > 0 and (self.step + 1) % info_dict['save_per_step'] == 0 and \
                   (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    dist.barrier()
                    self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=False)
                    model.train()
                if (batch_idx + 1) % info_dict["accum_grad"] == 0:
                    self.step += 1
        dist.barrier()
        self.cv(model, cv_data_loader, writer, info_dict, on_batch_end=True)

    @torch.inference_mode()
    def cv(self, model, cv_data_loader, writer, info_dict, on_batch_end=True):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} CV rank {}'.format(self.epoch, self.step + 1, on_batch_end, self.rank))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        for batch_idx, batch_dict in enumerate(cv_data_loader):
            info_dict["tag"] = "CV"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx

            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts

            if self.gan is True:
                batch_dict['turn'] = 'generator'
            info_dict = batch_forward(model, batch_dict, None, info_dict)

            for k, v in info_dict['loss_dict'].items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = []
                total_loss_dict[k].append(v.mean().item() * num_utts)
            log_per_step(None, info_dict)
        for k, v in total_loss_dict.items():
            total_loss_dict[k] = sum(v) / total_num_utts
        info_dict['loss_dict'] = total_loss_dict
        log_per_save(writer, info_dict)
        model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        save_model(model, model_name, info_dict)


    @torch.inference_mode()
    def generate(self, model, generate_data_loader, writer, info_dict, on_batch_end=True,hift=None, output_folder=None):
        ''' Cross validation on
        '''
        logging.info('Epoch {} Step {} on_batch_end {} Start Generating'.format(self.epoch, self.step + 1, on_batch_end))
        model.eval()
        total_num_utts, total_loss_dict = 0, {}  # avoid division by 0
        if output_folder==None:
            output_folder=info_dict['model_dir']
        for batch_idx, batch_dict in enumerate(generate_data_loader):
            print(batch_idx)
            info_dict["tag"] = "GENERATE"
            info_dict["step"] = self.step
            info_dict["epoch"] = self.epoch
            info_dict["batch_idx"] = batch_idx
            
            num_utts = len(batch_dict["utts"])
            total_num_utts += num_utts
            ref_wavs=batch_dict['wavs']
            speech_token=batch_dict['speech_token']
            speech_token_len=batch_dict['speech_token_len']
            speech_feat=batch_dict['speech_feat']
            speech_feat_len=batch_dict['speech_feat_len']
            speech_embedding=batch_dict['embedding']
            path=batch_dict['wavs'][0]
            name=os.path.splitext(os.path.basename(path))[0]
            random_ratios = torch.rand(1) * 0.5
            prompt_lengths = (speech_token_len.min().float() * random_ratios).int().to(speech_feat.device)
            prompt_token=speech_token[:,:prompt_lengths]
            input_token=speech_token[:,prompt_lengths:]
            input_token_len=speech_token_len-prompt_lengths
            prompt_token_len=speech_token_len-input_token_len
            prompt_feat_lengths=prompt_lengths*model.module.token_mel_ratio
            input_feat_len=speech_feat_len-prompt_feat_lengths
            prompt_feat_len=speech_feat_len-input_feat_len
            input_feat=speech_feat[:,prompt_feat_lengths:]
            prompt_feat=speech_feat[:,:prompt_feat_lengths]
            device=model.module.encoder_proj.weight.device
            mel=model.module.inference(input_token.to(device),input_token_len.to(device),prompt_token.to(device),prompt_token_len.to(device),prompt_feat.to(device),prompt_feat_len.to(device),speech_embedding.to(device),streaming=True,finalize=True)[0]
            mel=torch.cat([prompt_feat.to(mel.device).transpose(-1,-2),mel],dim=-1)
            gen_speech=hift.inference(mel)[0]
            ref_audio_source_path = ref_wavs[0]
            dataset_name = get_dataset_name_from_path(ref_audio_source_path)
            ref_audio_output_dir = os.path.join(output_folder, 'ref', dataset_name)
            os.makedirs(ref_audio_output_dir, exist_ok=True)
            ref_audio_dest_path = os.path.join(ref_audio_output_dir, f'{name}.wav')
            if not os.path.exists(ref_audio_dest_path):
                if ref_audio_source_path.lower().endswith('.wav'):
                    shutil.copy(ref_audio_source_path, ref_audio_dest_path)
                else:
                    waveform, sample_rate = torchaudio.load(ref_audio_source_path)
                    torchaudio.save(ref_audio_dest_path, waveform, sample_rate)
            generate_audio_output_dir = os.path.join(output_folder, 'generate', dataset_name)
            os.makedirs(generate_audio_output_dir, exist_ok=True)
            generated_audio_path = os.path.join(generate_audio_output_dir, f'{name}.wav')
            torchaudio.save(generated_audio_path, gen_speech.cpu(), 24000)
            
        #     if self.gan is True:
        #         batch_dict['turn'] = 'generator'
        #     info_dict = batch_forward(model, batch_dict, None, info_dict)

        #     for k, v in info_dict['loss_dict'].items():
        #         if k not in total_loss_dict:
        #             total_loss_dict[k] = []
        #         total_loss_dict[k].append(v.mean().item() * num_utts)
        #     log_per_step(None, info_dict)
        # for k, v in total_loss_dict.items():
        #     total_loss_dict[k] = sum(v) / total_num_utts
        # info_dict['loss_dict'] = total_loss_dict
        # log_per_save(writer, info_dict)
        # model_name = 'epoch_{}_whole'.format(self.epoch) if on_batch_end else 'epoch_{}_step_{}'.format(self.epoch, self.step + 1)
        # save_model(model, model_name, info_dict)