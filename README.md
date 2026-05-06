# LLM Fine-tune Toolkit

一键微调大模型：输入 PDF/MD 文件 + GGUF 模型 → 输出微调后的 GGUF 模型。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键微调
python finetune.py \
  --model /path/to/model.gguf \
  --input /path/to/pdfs_and_mds/ \
  --output /path/to/output.gguf \
  --epochs 1 \
  --samples 200
```

## 输入

- **模型**: GGUF 格式的模型文件 (如 Qwen3.5-9B-Q6_K.gguf)
- **数据**: PDF 文件和/或 Markdown 文件的目录

## 输出

- **GGUF 模型**: 微调后的 Q6_K 量化模型，可直接用于 llama.cpp / ollama

## 工作流程

```
PDF/MD 文件 → 提取文本 → 训练数据 → QLoRA 微调 → 合并 LoRA → GGUF Q6_K
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | GGUF 模型路径 |
| `--input` | 必填 | PDF/MD 文件目录 |
| `--output` | `./output.gguf` | 输出 GGUF 路径 |
| `--epochs` | 1 | 训练轮数 |
| `--samples` | 200 | 最大训练样本数 |
| `--seq-len` | 256 | 最大序列长度 |
| `--lora-r` | 8 | LoRA rank |
| `--batch-size` | 1 | 批次大小 |
| `--grad-accum` | 8 | 梯度累积步数 |
| `--lr` | 2e-4 | 学习率 |

## 系统要求

- Python 3.10+
- PyTorch 2.6+ (CUDA 12.4)
- 16GB+ VRAM (对于 9B 模型)
- llama.cpp (用于 GGUF 转换和量化)

## 示例

```bash
# 微调 Qwen3.5-9B 用于模拟集成电路设计
python finetune.py \
  --model /root/Qwen3.5-9B-DeepSeek-V4-Flash-Q6_K.gguf \
  --input /home/obsidian/mike/raw/ \
  --output /home/model/Qwen3.5-9B-analog-ic-Q6_K.gguf \
  --epochs 1 \
  --samples 200 \
  --system-prompt "你是模拟集成电路设计专家。"
```

## 文件结构

```
llm-finetune-toolkit/
├── finetune.py          # 主脚本
├── extract_data.py      # 数据提取
├── merge_and_export.py  # 合并和导出
├── requirements.txt     # 依赖
└── README.md            # 说明
```
