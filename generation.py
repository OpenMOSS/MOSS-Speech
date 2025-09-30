"""MossSpeech inference demo aligned with Hugging Face Transformers guidelines."""
import os
from dataclasses import astuple

import torch
import torchaudio

from transformers import (
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)


prompt = "为什么鸡蛋没有鸡？"
prompt_audio = "./assets/prompt.wav"
model_path = "/inspire/ssd/project/embodied-multimodality/feichaoye-p-feizhaoye/gaoyang/moss-transformers/moss-transformers/MossSpeech-SFT-fp32-v2.1-80K"
codec_path = "/inspire/ssd/project/embodied-multimodality/feichaoye-p-feizhaoye/gaoyang/moss-transformers/moss-transformers/MossSpeechCodec-HF"
output_path = "outputs"
output_modality = "audio"  # or text

generation_config = GenerationConfig(
    temperature=0.7,
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    max_new_tokens=1000,
    min_new_tokens=10,
    do_sample=True,
    use_cache=True,
)


class StopOnToken(StoppingCriteria):
    """Stop generation once the final token equals the provided stop ID."""

    def __init__(self, stop_id: int) -> None:
        super().__init__()
        self.stop_id = stop_id

    def __call__(self, input_ids: torch.LongTensor, scores) -> bool:  # type: ignore[override]
        return input_ids[0, -1].item() == self.stop_id


def prepare_stopping_criteria(processor):
    tokenizer = processor.tokenizer
    stop_tokens = [
        tokenizer.pad_token_id,
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    ]
    return StoppingCriteriaList([StopOnToken(token_id) for token_id in stop_tokens])


messages = [
    [
        {
            "role": "system",
            "content": "您好！我叫模思智能语音助手，可以帮您解答问题、提供信息和协助完成任务。您可以问我关于历史、科学、技术、娱乐等各类问题，我会尽力为您提供建议和帮助。请问有什么我可以帮您的吗？"},
        {
            "role": "user",
            "content": prompt
        }
    ]
]


processor = AutoProcessor.from_pretrained(model_path, codec_path=codec_path, device="cuda", trust_remote_code=True)
stopping_criteria = prepare_stopping_criteria(processor)
encoded_inputs = processor(messages, output_modality)

model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map="cuda").eval()

with torch.inference_mode():
    token_ids = model.generate(
        input_ids=encoded_inputs["input_ids"].to("cuda"),
        attention_mask=encoded_inputs["attention_mask"].to("cuda"),
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
    )

results = processor.decode(token_ids, output_modality, decoder_audio_prompt_path=prompt_audio)

os.makedirs(output_path, exist_ok=True)
for index, (result, modality) in enumerate(zip(results, output_modality)):
    audio, text, sample_rate = astuple(result)
    if modality == "audio":
        torchaudio.save(f"{output_path}/audio_{index}.wav", audio, sample_rate)
    else:
        print(text)
