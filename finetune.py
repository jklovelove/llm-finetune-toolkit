#!/usr/bin/env python3
"""
LLM Fine-tune Toolkit
一键微调大模型：PDF/MD → 训练数据 → QLoRA → 合并 → GGUF Q6_K
"""
import argparse
import os
import sys
import subprocess
import tempfile
import shutil

def find_llama_cpp():
    """Find llama.cpp binaries."""
    # Check common locations
    candidates = [
        os.path.expanduser("~/llama.cpp/build/bin"),
        "/usr/local/bin",
        os.path.expanduser("~/.local/bin"),
    ]
    for path in candidates:
        quantize = os.path.join(path, "llama-quantize")
        convert = os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py")
        if os.path.exists(quantize) and os.path.exists(convert):
            return path, convert
    return None, None

def extract_data(input_dir, output_file, max_samples=200):
    """Extract training data from PDF and MD files."""
    print(f"\n{'='*60}")
    print(f"Step 1: Extracting training data from {input_dir}")
    print(f"{'='*60}")

    import re
    import glob

    def clean_text(text):
        text = text.replace('\f', '\n')
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Authorized licensed use limited to:.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*Downloaded on.*$', '', text, flags=re.MULTILINE)
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        return text.strip()

    def extract_pdf(pdf_path):
        try:
            result = subprocess.run(
                ['pdftotext', '-layout', pdf_path, '-'],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0 and result.stdout.strip():
                return clean_text(result.stdout)
        except Exception as e:
            print(f"  ERROR: {e}")
        return None

    def read_markdown(md_path):
        try:
            with open(md_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read().strip()
        except Exception as e:
            print(f"  ERROR: {e}")
        return None

    pdf_files = sorted(glob.glob(os.path.join(input_dir, "**/*.pdf"), recursive=True))
    md_files = sorted(glob.glob(os.path.join(input_dir, "**/*.md"), recursive=True))

    print(f"Found {len(pdf_files)} PDFs and {len(md_files)} markdown files")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    total_chars = 0
    total_docs = 0

    with open(output_file, 'w', encoding='utf-8') as out:
        for i, pdf_path in enumerate(pdf_files):
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  [{i+1}/{len(pdf_files)}] Processing PDFs...")
            text = extract_pdf(pdf_path)
            if text and len(text) > 100:
                rel_path = os.path.relpath(pdf_path, input_dir)
                out.write(f"\n\n### Document: {rel_path}\n\n{text}\n")
                total_chars += len(text)
                total_docs += 1

        for i, md_path in enumerate(md_files):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(md_files)}] Processing markdown...")
            text = read_markdown(md_path)
            if text and len(text) > 50:
                rel_path = os.path.relpath(md_path, input_dir)
                out.write(f"\n\n### Document: {rel_path}\n\n{text}\n")
                total_chars += len(text)
                total_docs += 1

    print(f"  Documents: {total_docs}, Characters: {total_chars:,}")
    print(f"  Output: {output_file}")
    return output_file

def prepare_training_data(corpus_file, output_dir, max_samples=200, system_prompt="你是AI助手。"):
    """Convert raw text to chat format training data."""
    print(f"\n{'='*60}")
    print(f"Step 2: Preparing training data")
    print(f"{'='*60}")

    import json
    import random
    random.seed(42)

    examples = []
    with open(corpus_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    documents = content.split("### Document:")
    prompts = ["请解释以下概念：", "请分析以下文档：", "请总结以下内容："]

    for doc in documents[1:]:
        doc = doc.strip()
        if len(doc) < 200:
            continue
        lines = doc.split('\n')
        title = lines[0].strip()[:80]
        chunk = '\n'.join(l for l in lines[1:] if l.strip())[:400]
        if len(chunk) < 100:
            continue
        examples.append({
            "conversations": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{random.choice(prompts)}\n\n{title}\n\n{chunk}"},
                {"role": "assistant", "content": f"根据文档「{title}」：\n\n{chunk}"}
            ]
        })
        if len(examples) >= max_samples:
            break

    train_file = os.path.join(output_dir, "train_data.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    with open(train_file, 'w') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f"  Created {len(examples)} training examples")
    print(f"  Output: {train_file}")
    return train_file

def run_qlora_training(model_path, train_file, output_dir, args):
    """Run QLoRA fine-tuning on GPU."""
    print(f"\n{'='*60}")
    print(f"Step 3: QLoRA fine-tuning")
    print(f"{'='*60}")

    import torch
    import types
    import importlib.metadata

    # Patch torchao
    _orig_version = importlib.metadata.version
    def _patched_version(name):
        if name == 'torchao':
            raise importlib.metadata.PackageNotFoundError
        return _orig_version(name)
    importlib.metadata.version = _patched_version

    def _make_dummy(name):
        m = types.ModuleType(name)
        m.__version__ = '0.0.0'
        m.__spec__ = importlib.util.spec_from_loader(name, loader=None)
        m.__path__ = []
        return m

    for n in ['torchao', 'torchao.quantization', 'torchao.prototype',
              'torchao.prototype.safetensors', 'torchao.prototype.safetensors.safetensors_support']:
        sys.modules[n] = _make_dummy(n)

    # Patch bitsandbytes
    import bitsandbytes.nn.modules as _bnb_mod
    _orig_new = _bnb_mod.Params4bit.__new__
    def _patched_new(cls, *args, **kwargs):
        kwargs.pop('_is_hf_initialized', None)
        return _orig_new(cls, *args, **kwargs)
    _bnb_mod.Params4bit.__new__ = staticmethod(_patched_new)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    # Find HF model directory (from GGUF)
    # We need to download the HF version
    print(f"  Loading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f}GB allocated")

    # Apply LoRA
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()

    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            targets.add(name.split('.')[-1])
    lora_targets = [t for t in ['q_proj', 'v_proj', 'k_proj', 'o_proj'] if t in targets]
    print(f"  LoRA targets: {lora_targets}")

    lora_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        target_modules=lora_targets, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train
    dataset = load_dataset("json", data_files=train_file, split="train")

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        logging_steps=1,
        save_steps=50,
        save_total_limit=1,
        max_seq_length=args.seq_len,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        remove_unused_columns=False,
        max_grad_norm=0.3,
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(model=model, args=sft_config, train_dataset=dataset, processing_class=tokenizer)
    print("  Training started...")
    trainer.train()
    print("  Training completed!")

    # Save adapter
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved adapter to {output_dir}")
    return output_dir

def merge_lora(base_model_path, adapter_path, output_path):
    """Merge LoRA adapter into base model."""
    print(f"\n{'='*60}")
    print(f"Step 4: Merging LoRA into base model")
    print(f"{'='*60}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16,
        trust_remote_code=True, device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print(f"  Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    print(f"  Merging...")
    model = model.merge_and_unload()

    print(f"  Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"  Saved to {output_path}")
    return output_path

def convert_to_gguf(hf_model_path, output_gguf_path, llama_cpp_dir=None):
    """Convert HF model to GGUF Q6_K."""
    print(f"\n{'='*60}")
    print(f"Step 5: Converting to GGUF Q6_K")
    print(f"{'='*60}")

    if llama_cpp_dir is None:
        llama_cpp_dir = os.path.expanduser("~/llama.cpp")

    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    quantize_bin = os.path.join(llama_cpp_dir, "build/bin/llama-quantize")

    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}")
    if not os.path.exists(quantize_bin):
        raise FileNotFoundError(f"llama-quantize not found at {quantize_bin}")

    # Convert to f16 GGUF
    f16_gguf = output_gguf_path.replace(".gguf", "-f16.gguf")

    print(f"  Converting HF → GGUF f16...")
    subprocess.run([
        sys.executable, convert_script,
        hf_model_path,
        "--outfile", f16_gguf,
        "--outtype", "f16",
    ], check=True)

    print(f"  Quantizing to Q6_K...")
    subprocess.run([
        quantize_bin, f16_gguf, output_gguf_path, "Q6_K",
    ], check=True)

    # Cleanup
    os.remove(f16_gguf)
    print(f"  Output: {output_gguf_path}")
    return output_gguf_path

def main():
    parser = argparse.ArgumentParser(
        description="LLM Fine-tune Toolkit: PDF/MD → 微调 GGUF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="GGUF 模型路径或 HF 模型路径")
    parser.add_argument("--input", required=True, help="PDF/MD 文件目录")
    parser.add_argument("--output", default="./output.gguf", help="输出 GGUF 路径")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--samples", type=int, default=200, help="最大训练样本数")
    parser.add_argument("--seq-len", type=int, default=256, help="最大序列长度")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=1, help="批次大小")
    parser.add_argument("--grad-accum", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--system-prompt", default="你是AI助手。", help="系统提示词")
    parser.add_argument("--work-dir", default="./work", help="工作目录")
    parser.add_argument("--llama-cpp-dir", default=os.path.expanduser("~/llama.cpp"), help="llama.cpp 目录")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练（仅提取数据）")
    parser.add_argument("--hf-model", help="HF 模型路径（如果 --model 是 GGUF）")

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    # Step 1: Extract data
    corpus_file = os.path.join(args.work_dir, "training_corpus.txt")
    extract_data(args.input, corpus_file, args.samples)

    if args.skip_training:
        print("\nSkipping training (data extraction only)")
        return

    # Step 2: Prepare training data
    train_file = prepare_training_data(
        corpus_file, args.work_dir, args.samples, args.system_prompt
    )

    # Step 3: QLoRA training
    # Need HF model path
    hf_model = args.hf_model or args.model
    adapter_dir = os.path.join(args.work_dir, "lora_adapter")
    run_qlora_training(hf_model, train_file, adapter_dir, args)

    # Step 4: Merge LoRA
    merged_dir = os.path.join(args.work_dir, "merged_model")
    merge_lora(hf_model, adapter_dir, merged_dir)

    # Step 5: Convert to GGUF
    convert_to_gguf(merged_dir, args.output, args.llama_cpp_dir)

    print(f"\n{'='*60}")
    print(f"Done! Output: {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
