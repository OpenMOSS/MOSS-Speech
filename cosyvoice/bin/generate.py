# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
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

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import os
import torch
import torch.distributed as dist
import deepspeed

from hyperpyyaml import load_hyperpyyaml

from torch.distributed.elastic.multiprocessing.errors import record
from cosyvoice.utils.losses import DPOLoss
from cosyvoice.utils.executor import Executor
from cosyvoice.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--ref_model', required=False, help='ref model used in dpo')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--generate_data', required=True, help='generate data file')
    parser.add_argument('--qwen_pretrain_path', required=False, help='qwen pretrain path')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--hift_checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--use_amp',
                        action='store_true',
                        default=False,
                        help='Use automatic mixed precision training')
    parser.add_argument('--dpo',
                        action='store_true',
                        default=False,
                        help='Use Direct Preference Optimization')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--validate_interval',
                        default=20000)
    parser.add_argument('--timeout',
                        default=600000,
                        type=int,
                        help='timeout (in seconds) of cosyvoice_join.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # gan train has some special initialization logic
    gan = True if args.model == 'hifigan' else False

    override_dict = {k: None for k in ['llm', 'flow'] if k != args.model}
    if gan is True:
        override_dict.pop('hift')
    try:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides={**override_dict, 'qwen_pretrain_path': args.qwen_pretrain_path})
    except Exception:
        with open(args.config, 'r') as f:
            configs = load_hyperpyyaml(f, overrides=override_dict)
    if gan is True:
        configs['train_conf'] = configs['train_conf_gan']
    configs['train_conf'].update(vars(args))

    # Init env for ddp
    init_distributed(args)
    logging.info(f"Successfully init distributed")
    # Get dataset & dataloader
    train_dataset, cv_dataset, generate_dataset,train_data_loader, cv_data_loader,generate_data_loader = \
        init_dataset_and_dataloader(args, configs, gan, args.dpo)
    
    logging.info(f"Successfully init dataset and dataloader")
    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)
    logging.info(f"Successfully check_modify_and_save_config")
    # Tensorboard summary
    writer = init_summarywriter(args)
    logging.info(f"Successfully init_summarywriter")
    # load checkpoint
    if args.dpo is True:
        configs[args.model].forward = configs[args.model].forward_dpo
    model = configs[args.model]
    logging.info(f"Successfully init_model")
    start_step, start_epoch = 0, -1
    if args.checkpoint is not None:
        if os.path.exists(args.checkpoint):
            checkpoint_state_dict = torch.load(args.checkpoint, map_location='cpu')
            state_dict=checkpoint_state_dict
            #model.load_state_dict(state_dict, strict=False)
            model_state_dict = model.state_dict()
            filtered_state_dict = {}
            loaded_keys = []
            skipped_keys = []

            for k, v in checkpoint_state_dict.items():
                if isinstance(v,int):
                    continue
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                    loaded_keys.append(k)
                else:
                    skipped_keys.append(k)
            model_state_dict.update(filtered_state_dict)
            model.load_state_dict(model_state_dict, strict=True)
            if skipped_keys:
                logging.warning("Weights for the following keys were SKIPPED due to name/shape mismatch:")
                for key in skipped_keys:
                    shape_in_checkpoint = checkpoint_state_dict[key].shape if key in checkpoint_state_dict else 'N/A'
                    shape_in_model = model_state_dict[key].shape if key in model_state_dict else 'N/A'
                    logging.warning(f"  - {key} (Checkpoint: {shape_in_checkpoint}, Model: {shape_in_model})")
            logging.info(f"Successfully loaded {len(loaded_keys)} matching parameters.")
            # if 'step' in checkpoint_state_dict:
            #     start_step = checkpoint_state_dict['step']
            # if 'epoch' in checkpoint_state_dict:
            #     start_epoch = checkpoint_state_dict['epoch']
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))

    if args.hift_checkpoint is not None:
        hift=configs['hift']
        if os.path.exists(args.hift_checkpoint):
            state_dict = torch.load(args.hift_checkpoint, map_location='cpu')
            hift.load_state_dict(state_dict, strict=True)
            hift.eval()
            for param in hift.parameters():
                param.requires_grad = False
                
            logging.info("Hift model loaded in inference mode (eval, no_grad).")
            if 'step' in state_dict:
                start_step = state_dict['step']
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch']
        else:
            logging.warning('checkpoint {} do not exsist!'.format(args.checkpoint))
        
    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)
    hift=hift.cuda()
    # Get optimizer & scheduler
    model, optimizer, scheduler, optimizer_d, scheduler_d = init_optimizer_and_scheduler(args, configs, model, gan)

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])
    info_dict['step'] = start_step
    info_dict['epoch'] = start_epoch

    # DPO related
    if args.dpo is True:
        ref_model = deepcopy(configs[args.model])
        state_dict = torch.load(args.ref_model, map_location='cpu')
        ref_model.load_state_dict(state_dict, strict=False)
        dpo_loss = DPOLoss(beta=0.01, label_smoothing=0.0, ipo=False)
        # NOTE maybe it is not needed to wrap ref_model as ddp because its parameter is not updated
        ref_model = wrap_cuda_model(args, ref_model)
    else:
        ref_model, dpo_loss = None, None

    # Get executor
    executor = Executor(gan=gan, ref_model=ref_model, dpo_loss=dpo_loss)
    executor.step = start_step
    # Start training loop
    executor.epoch = 0
    dist.barrier()
    executor.generate(model,generate_data_loader, writer, info_dict,hift=hift)


if __name__ == '__main__':
    main()
