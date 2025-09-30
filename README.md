# MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance

<div align="center" style="line-height: 1;">
    <!-- <a href="https://open-moss.com/cn/speechgpt2-preview/" target="_blank" style="margin: 2px;">
        <img alt="Project Page" src="https://img.shields.io/badge/🏠%20Project%20Page-MOSS--Speech-536af5?color=e31a2f&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://sp2.open-moss.com/" target="_blank" style="margin: 2px;">
        <img alt="Chat" src="https://img.shields.io/badge/🤖%20Demo-MOSS--Speech-536af5?color=1ae3f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>  -->
    <a href="https://moss-speech.open-moss.com/" target="_blank" style="margin: 2px;">
    <img alt="Video Demo" src="https://img.shields.io/badge/📹%20Video%20Demo-MOSS--Speech-536af5?color=1ae3f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="papers/MOSS-Speech Technical Report.pdf" target="_blank" style="margin: 2px;">
    <img alt="Technical Report" src="https://img.shields.io/badge/📄%20Technical%20Report-MOSS--Speech-4caf50?color=4caf50&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://discord.gg/wmJGnd4q" target="_blank" style="margin: 2px;">
        <img alt="Discord" src="https://img.shields.io/badge/Discord-OpenMOSS-7289da?logo=discord&logoColor=white&color=7289da" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://huggingface.co/fnlp" target="_blank" style="margin: 2px;">
        <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-MOSS--Speech-ffc107?color=ffc107&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
    <a href="https://x.com/Open_MOSS" target="_blank" style="margin: 2px;">
    <img alt="X Follow" src="https://img.shields.io/badge/Twitter-OpenMOSS-black?logo=x&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>
</div>

<div style="height: 200px; overflow: hidden; text-align:center;">
    <img src="assets/logo-large.png" style="width:80%; object-fit:cover; object-position:center;">
</div>



阅读[中文](./README_ZH.md)版本.

---

## 📖 Introduction

Spoken dialogue systems often rely on cascaded pipelines that transcribe, process, and resynthesize speech, which limits expressivity and discards paralinguistic cues. **MOSS-Speech** directly understands and generates speech without relying on text intermediates, enabling end-to-end speech interaction while preserving tone, prosody, and emotion.  

Our approach combines a **modality-based layer-splitting architecture** with a **frozen pre-training strategy**, leveraging pretrained text LLMs while extending native speech capabilities. Experiments show state-of-the-art results in spoken question answering and competitive speech-to-speech performance compared to text-guided systems.  

<!-- Welcome to talk to our [Demo system](https://sp2.open-moss.com/) online, and also welcome to check out the system's [demonstration video](https://open-moss.com/en/speechgpt2-preview/).-->

Welcome to check out the system's [demonstration video](https://moss-speech.open-moss.com/).



---

## 🔑 Key Features


- **True Speech-to-Speech Modeling**: No text guidance required.  
- **Layer-Splitting Architecture**: Integrates modality-specific layers on top of pretrained text LLM backbones.  
- **Frozen Pre-Training Strategy**: Preserves LLM reasoning while enhancing speech understanding and generation.  
- **State-of-the-Art Performance**: Excels in spoken question answering and speech-to-speech tasks.  
- **Expressive & Efficient**: Maintains paralinguistic cues often lost in cascaded pipelines, such as tone, emotion, and prosody.  

---

## 📂 Repository Contents

- `gradio_demo.py` – Gradio-based web demo script for quickly experiencing speech-to-speech functionality.  
- `generation.py` – Core generation script for producing output speech from input speech, suitable for inference and batch processing.  

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/OpenMOSS/MOSS-Speech
cd MOSS-Speech

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage
Launch the web demo
```sh
python3 gradio_demo.py
```

<p align="center">
    <img src="assets/gradio.jpg" width="80%"> <br>
</p>


---

## License
- The code in this repository is released under the [Apache 2.0](LICENSE) license.

---

## Acknowledgements
- [Qwen](https://github.com/QwenLM/Qwen3): We use Qwen3-8B-Instruct as the base model.
- We thank an anonymous colleague for Character Voice!

---

## 📜 Citation

If you use this repository or model in your research, please cite:

```bibtex
@article{moss_speech2025,
  title={MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance},
  author={SLM Team},
  institution={Shanghai Innovation Institute, Fudan University, MOSI},
  year={2025},
  note={Official implementation available at https://huggingface.co/fnlp/MOSS-Speech}
}

or

@misc{moss_speech2025,
  author = {SLM Team},
  title = {MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/OpenMOSS/MOSS-Speech}},
}
```