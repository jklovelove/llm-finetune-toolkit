#!/usr/bin/env python3
"""Test the fine-tuned LoRA adapter."""
import os
import sys
import types

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Patch torchao
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

# Patch bitsandbytes
import bitsandbytes.nn.modules as _bnb_mod
_orig_new = _bnb_mod.Params4bit.__new__
def _patched_new(cls, *args, **kwargs):
    kwargs.pop('_is_hf_initialized', None)
    return _orig_new(cls, *args, **kwargs)
_bnb_mod.Params4bit.__new__ = staticmethod(_patched_new)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_ID = "/root/Qwen3.5-9B"
ADAPTER_PATH = "/home/model/lora_output_9b"

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, low_cpu_mem_usage=True,
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

# Test prompt
prompt = "请解释以下模拟集成电路设计概念：\n\n运算放大器的共模抑制比（CMRR）"
messages = [
    {"role": "system", "content": "你是模拟集成电路设计专家。"},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("\n=== Test Output ===")
with torch.no_grad():
    outputs = model.generate(
        **inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9,
    )
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
