# Ascend环境修改说明
由于 Ascend 对部分操作不支持，在 Ascend 芯片上使用`torch_npu`推理时需要对部分代码进行修改，具体修改如下：

## 1. huggingface下载方式修改
在环境依赖中，对`diffusers/utils/dynamic_modules_utils.py`进行修改

由于对`cached_download`不支持，对第28行注释，并做以下修改
```python
#from huggingface_hub import cached_download, hf_hub_download, model_info
from huggingface_hub import hf_hub_download, model_info
```
此外，将288行的`cached_download`修改为`hf_hub_download`

## 2. Whisper特征提取修改
在环境依赖中，对`transformers/models/whisper/feature_extraction_whisper.py`进行修改

将第319行修改为:
```python
#input_features = extract_fbank_features(input_features[0], device)
input_features = extract_fbank_features(input_features[0], 'cpu')
```

## 3. Matcha-TTS Transformer模型推理修改
对`MOSS-Speech/Matcha-TTS/matcha/models/components/transformer.py`进行修改

第267行增加对bf16的转换：
```python
norm_hidden_states.to(torch.bfloat16)
```

## 4. 反向傅里叶变换修改
对`MOSS-Speech/cosyvoice/hifigan/generator.py`进行修改
```python
#inverse_transform = torch.istft(torch.complex(real, img), self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window.to(magnitude.device))
inverse_transform = torch.istft(torch.complex(real, img).to("cpu"), self.istft_params["n_fft"], self.istft_params["hop_len"], self.istft_params["n_fft"], window=self.stft_window.to("cpu"))
```