#!/usr/bin/env python3
import os
import json
import subprocess

import torch
import mlflow
from datasets import Dataset as HFDataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# —— 环境配置 —— #
os.environ["MLFLOW_TRACKING_URI"]      = "http://129.114.108.60:8000"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"

# —— 数据加载函数 —— #
def load_dataset(path: str) -> HFDataset:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return HFDataset.from_dict({
        "question": [item["question"] for item in data],
        "context":  [item["context"]  for item in data],
        "answer":   [item["answer"]   for item in data],
    })

# —— 文本预处理器 —— #
class Preprocessor:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, examples):
        input_ids, attention_mask, labels = [], [], []
        for q, c, a in zip(examples["question"], examples["context"], examples["answer"]):
            prompt_ids   = self.tokenizer(f"问题：{q}\n上下文：{c}\n回答：",
                                          add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(a + self.tokenizer.eos_token,
                                          add_special_tokens=False)["input_ids"]
            ids    = (prompt_ids + response_ids)[:self.max_length]
            labs   = ([-100]*len(prompt_ids) + response_ids)[:self.max_length]
            padlen = self.max_length - len(ids)
            ids    += [self.tokenizer.pad_token_id] * padlen
            labs   += [-100] * padlen
            mask   = [1]*len(ids[:-padlen]) + [0]*padlen
            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(labs)
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels}

def main():
    # —— 开始 MLflow run —— #
    mlflow.set_experiment("Commit QA Training")
    with mlflow.start_run():
        # —— 记录 GPU 信息 —— #
        for cmd in ["nvidia-smi", "rocm-smi -v"]:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                mlflow.log_text(r.stdout, "gpu-info.txt")
                break

        # —— 设备 & dtype —— #
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype  = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Using device={device}, dtype={dtype}")

        # —— Tokenizer & 4-bit 量化配置 —— #
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer  = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, padding_side="right")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

        # —— 量化加载模型 —— #
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=dtype,
            use_auth_token=True,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # —— 注入 LoRA adapter —— #
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # —— 加载并预处理数据 —— #
        ds = load_dataset("qa_from_commits_formatted.json")
        proc = Preprocessor(tokenizer, max_length=128)
        ds  = ds.map(proc, batched=True, batch_size=4, remove_columns=["question","context","answer"], num_proc=4)
        ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

        # —— 训练配置 —— #
        args = TrainingArguments(
            output_dir="/mnt/object/data/llama3_qa_4bit_lora",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            learning_rate=2e-5,
            bf16=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            logging_steps=100,
            save_steps=500,
            report_to="mlflow",
            dataloader_num_workers=2,
            dataloader_drop_last=True,
        )
        mlflow.log_params({
            "model": model_name,
            "max_length": 128,
            "batch_size": args.per_device_train_batch_size,
            "accum_steps": args.gradient_accumulation_steps,
            "lr": args.learning_rate,
            "lora_r": lora_cfg.r,
        })

        # —— 启动训练 —— #
        trainer = Trainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)
        trainer.train()

        # —— 保存模型 —— #
        model.save_pretrained(args.output_dir)
        print(f"Model and LoRA adapter saved at {args.output_dir}")

if __name__ == "__main__":
    main()
