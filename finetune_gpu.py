#!/usr/bin/env python3
"""
QLoRA fine-tuning of Qwen3.5-9B on GPU (4-bit quantization).
"""
import os
import sys
import types
import gc
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Patch torchao (incompatible with torch 2.6.0)
import importlib.metadata, importlib.util
_orig_version = importlib.metadata.version
def _patched_version(name):
    if name == 'torchao': raise importlib.metadata.PackageNotFoundError
    return _orig_version(name)
importlib.metadata.version = _patched_version
def _make_dummy(name):
    m = types.ModuleType(name); m.__version__ = '0.0.0'
    m.__spec__ = importlib.util.spec_from_loader(name, loader=None); m.__path__ = []
    return m
for n in ['torchao','torchao.quantization','torchao.prototype',
          'torchao.prototype.safetensors','torchao.prototype.safetensors.safetensors_support']:
    sys.modules[n] = _make_dummy(n)

# Patch bitsandbytes Params4bit
import bitsandbytes.nn.modules as _bnb_mod
_orig_new = _bnb_mod.Params4bit.__new__
def _patched_new(cls, *args, **kwargs):
    kwargs.pop('_is_hf_initialized', None)
    return _orig_new(cls, *args, **kwargs)
_bnb_mod.Params4bit.__new__ = staticmethod(_patched_new)

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

MODEL_ID = "/root/Qwen3.5-9B"
TRAINING_DATA = "/home/model/training_data/training_corpus.txt"
OUTPUT_DIR = "/home/model/lora_output_9b"
LORA_R = 8
LORA_ALPHA = 16
MAX_SEQ_LEN = 256
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
MAX_SAMPLES = 200

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Step 1: Prepare data
# ============================================================
print("\n=== Step 1: Preparing data ===")
import json, random
random.seed(42)

examples = []
with open(TRAINING_DATA, 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

documents = content.split("### Document:")
prompts = ["请解释以下模拟集成电路设计概念：", "请分析以下技术文档：", "请总结以下设计要点："]

for doc in documents[1:]:
    doc = doc.strip()
    if len(doc) < 200: continue
    lines = doc.split('\n')
    title = lines[0].strip()[:80]
    chunk = '\n'.join(l for l in lines[1:] if l.strip())[:400]
    if len(chunk) < 100: continue
    examples.append({
        "conversations": [
            {"role": "system", "content": "你是模拟集成电路设计专家。"},
            {"role": "user", "content": f"{random.choice(prompts)}\n\n{title}\n\n{chunk}"},
            {"role": "assistant", "content": f"根据文档「{title}」：\n\n{chunk}"}
        ]
    })
    if len(examples) >= MAX_SAMPLES: break

print(f"Created {len(examples)} examples")
train_file = os.path.join(OUTPUT_DIR, "train_data.jsonl")
with open(train_file, 'w') as f:
    for ex in examples: f.write(json.dumps(ex, ensure_ascii=False) + '\n')

# ============================================================
# Step 2: Load model with 4-bit quantization
# ============================================================
print("\n=== Step 2: Loading model (4-bit QLoRA) ===")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"Model loaded in {time.time()-t0:.0f}s")
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB allocated, {torch.cuda.memory_reserved() / 1e9:.1f}GB reserved")
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Step 3: Apply LoRA
# ============================================================
print("\n=== Step 3: Applying LoRA ===")
model = prepare_model_for_kbit_training(model)
model.enable_input_require_grads()

targets = set()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        targets.add(name.split('.')[-1])
lora_targets = [t for t in ['q_proj', 'v_proj', 'k_proj', 'o_proj'] if t in targets]
print(f"LoRA targets: {lora_targets}")

lora_config = LoraConfig(
    r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
    target_modules=lora_targets, bias="none", task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# Step 4: Train
# ============================================================
print("\n=== Step 4: Training ===")
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("json", data_files=train_file, split="train")

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR, num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE, lr_scheduler_type="cosine",
    warmup_steps=5, logging_steps=1, save_steps=50, save_total_limit=1,
    max_seq_length=MAX_SEQ_LEN, fp16=False, bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none", remove_unused_columns=False,
    max_grad_norm=0.3, dataloader_pin_memory=False,
)

trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer)
print("Training started...")
t0 = time.time()
trainer.train()
elapsed = time.time() - t0
print(f"\nTraining completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

print("\n=== Save ===")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Saved to {OUTPUT_DIR}")
