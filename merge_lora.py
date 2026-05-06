#!/usr/bin/env python3
"""Merge LoRA adapter into base model."""
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/root/Qwen3.5-9B"
ADAPTER_PATH = "/home/model/lora_output_9b"
OUTPUT_PATH = "/home/model/Qwen3.5-9B-analog-ic"

print("Loading base model (float16)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.float16, trust_remote_code=True, device_map="cpu",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)

print("Merging LoRA into base weights...")
model = model.merge_and_unload()

print(f"Saving merged model to {OUTPUT_PATH}...")
model.save_pretrained(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print("Done!")
