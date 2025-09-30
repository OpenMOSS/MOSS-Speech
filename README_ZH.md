# MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance

<div align="center" style="line-height: 1;">
    <!-- <a href="https://open-moss.com/cn/speechgpt2-preview/" target="_blank" style="margin: 2px;">
        <img alt="Project Page" src="https://img.shields.io/badge/🏠%20Project%20Page-MOSS--Speech-536af5?color=e31a2f&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a> 
    <a href="https://sp2.open-moss.com/" target="_blank" style="margin: 2px;">
        <img alt="Chat" src="https://img.shields.io/badge/🤖%20Demo-MOSS--Speech-536af5?color=1ae3f5&logoColor=white" style="display: inline-block; vertical-align: middle;"/>
    </a>-->
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


Read this in [English](./README.md).

---

## 📖 介绍

语音对话系统通常依赖于级联式流水线，将语音先转录、处理，再重新合成，这种设计限制了表达能力，并丢失了副语言信息。**MOSS-Speech** 能够直接理解和生成语音，无需依赖文本中间表示，实现端到端的语音交互，同时保留语调、韵律和情感信息。  

我们的方法结合了 **基于模态的层拆分架构** 与 **冻结预训练策略**，在利用预训练文本大型语言模型的推理与知识能力的同时，扩展了原生语音处理能力。实验结果显示，该模型在语音问答任务上取得了最先进的性能，并在语音到语音生成任务中，相较于文本引导系统仍保持竞争力。  

<!--欢迎在线体验我们的[Demo系统](https://sp2.open-moss.com/)，也欢迎查看系统的[演示视频](https://open-moss.com/cn/speechgpt2-preview/)。-->
欢迎查看我们系统的[演示视频](https://moss-speech.open-moss.com/)。

---

## 🔑 核心特性

- **真正的语音到语音建模**：无需文本引导。  
- **层拆分架构**：在预训练文本 LLM 的基础上整合模态特定层。  
- **冻结预训练策略**：保留 LLM 推理能力，同时增强语音理解和生成能力。  
- **领先性能**：在语音问答和语音到语音任务中表现出色。  
- **表达丰富且高效**：保留流水线中常丢失的副语言信息（如语调、情感、韵律）。  

---

## 📂 仓库内容

- `gradio_demo.py` – 基于 Gradio 的在线演示脚本，用于快速体验语音到语音模型的功能。  
- `generation.py` – 核心生成脚本，用于从输入语音生成输出语音，可作为推理和批量处理工具。 

---

## 🛠️ 安装

```bash
# Clone the repository
git clone https://github.com/OpenMOSS/MOSS-Speech
cd MOSS-Speech

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 使用
### 启动网页demo

```sh
python3 gradio_demo.py
```

<p align="center">
    <img src="assets/gradio.jpg" width="80%"> <br>
</p>


---

## 协议
- 本开源仓库的代码遵循 [Apache 2.0](LICENSE) 协议。

---

## 致谢
- [Qwen](https://github.com/QwenLM/Qwen3): 我们以Qwen3-8B-Instruct作为基座模型。
- 感谢一位匿名的同事给我们提供声音!

---

## 📜 引用

如果在研究中使用本仓库或模型，请引用如下文献：

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