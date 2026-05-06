# LLM Fine-tune Toolkit

一键微调大模型：输入 PDF/MD 文件 + GGUF 模型 → 输出微调后的 GGUF 模型。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 一键微调
python finetune.py \
  --model /path/to/model.gguf \
  --hf-model /path/to/hf-model/ \
  --input /path/to/pdfs_and_mds/ \
  --output /path/to/output.gguf \
  --epochs 1 \
  --samples 200
```

## 输入

- **模型**: GGUF 格式或 HuggingFace 格式的模型
- **数据**: PDF 文件和/或 Markdown 文件的目录

## 输出

- **GGUF 模型**: 微调后的 Q6_K 量化模型，可直接用于 llama.cpp / ollama

## 工作流程

```
PDF/MD 文件 → 提取文本 → 训练数据 → QLoRA 微调 → 合并 LoRA → GGUF Q6_K
```

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `finetune.py` | 主脚本，一键完成全流程 |
| `extract_data.py` | 从 PDF/MD 提取训练语料 |
| `finetune_gpu.py` | QLoRA 微调（GPU，4-bit 量化） |
| `merge_lora.py` | 将 LoRA adapter 合并到基础模型 |
| `test_model.py` | 测试微调后的模型 |

## 分步执行

如果不想用一键脚本，可以分步执行：

```bash
# Step 1: 提取数据
python extract_data.py --input /path/to/pdfs/ --output training_corpus.txt

# Step 2: QLoRA 微调
python finetune_gpu.py \
  --model /path/to/hf-model/ \
  --data training_corpus.txt \
  --output ./lora_adapter/ \
  --epochs 1 \
  --samples 200

# Step 3: 合并 LoRA
python merge_lora.py \
  --base-model /path/to/hf-model/ \
  --adapter ./lora_adapter/ \
  --output ./merged_model/

# Step 4: 转换为 GGUF（需要 llama.cpp）
python convert_to_gguf.py --input ./merged_model/ --output model.gguf

# Step 5: 测试
python test_model.py --model ./merged_model/ --adapter ./lora_adapter/
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | 必填 | GGUF 模型路径 |
| `--hf-model` | 必填 | HuggingFace 模型路径 |
| `--input` | 必填 | PDF/MD 文件目录 |
| `--output` | `./output.gguf` | 输出 GGUF 路径 |
| `--epochs` | 1 | 训练轮数 |
| `--samples` | 200 | 最大训练样本数 |
| `--seq-len` | 256 | 最大序列长度 |
| `--lora-r` | 8 | LoRA rank |
| `--batch-size` | 1 | 批次大小 |
| `--grad-accum` | 8 | 梯度累积步数 |
| `--lr` | 2e-4 | 学习率 |
| `--system-prompt` | "你是AI助手。" | 系统提示词 |

## 训练时间参考（V100-16GB，9B 模型）

| 样本数 | 步数 | 时间 |
|--------|------|------|
| 50 | 7 | ~4 min |
| 200 | 25 | ~14 min |
| 500 | 63 | ~35 min |

## 系统要求

- Python 3.10+
- PyTorch 2.6+ (CUDA 12.4)
- 16GB+ VRAM（9B 模型）
- llama.cpp（用于 GGUF 转换和量化）

## 示例

```bash
# 微调 Qwen3.5-9B 用于模拟集成电路设计
python finetune.py \
  --model /root/Qwen3.5-9B-DeepSeek-V4-Flash-Q6_K.gguf \
  --hf-model /root/Qwen3.5-9B/ \
  --input /home/obsidian/mike/raw/ \
  --output /home/model/Qwen3.5-9B-analog-ic-Q6_K.gguf \
  --epochs 1 \
  --samples 200 \
  --system-prompt "你是模拟集成电路设计专家。"
```
