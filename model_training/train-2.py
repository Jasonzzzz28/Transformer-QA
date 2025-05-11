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

# —— 1. 加载原始 JSON 数据 —— #
def load_dataset(file_path: str) -> HFDataset:
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return HFDataset.from_dict({
        "question": [item["question"] for item in raw],
        "context":  [item["context"]  for item in raw],
        "answer":   [item["answer"]   for item in raw],
    })

# —— 2. 定义 Preprocessor —— #
class LlamaQAPreprocessor:
    def __init__(self, tokenizer, max_length=128):  # —↓ 再减 max_length
        self.tokenizer = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, examples):
        input_ids_batch, attention_mask_batch, labels_batch = [], [], []
        for q, c, a in zip(examples["question"], examples["context"], examples["answer"]):
            prompt = f"### 问题：{q}\n### 上下文：{c}\n### 回答："
            prompt_ids   = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(a + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids    = (prompt_ids + response_ids)[: self.max_length]
            labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]
            attention_mask = [1] * len(ids)
            pad_len = self.max_length - len(ids)
            if pad_len > 0:
                ids += [self.tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len
            input_ids_batch.append(ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)
        return {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch, "labels": labels_batch}

# —— 3. 设备选择 —— #
def setup_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.9)
        return "mps"
    else:
        return "cpu"

if __name__ == "__main__":
    # 防止显存碎片化
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"
    os.environ["MLFLOW_TRACKING_URI"] = "http://129.114.108.60:8000"

    mlflow.set_experiment("Commit QA Training - Llama3.1-Instruct-4bit")
    try:
        mlflow.end_run()
    except:
        pass

    with mlflow.start_run(log_system_metrics=True):
        # —— 4. 记录 GPU/CPU 信息 —— #
        gpu_info = None
        for cmd in ["nvidia-smi", "rocm-smi -v"]:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                gpu_info = r.stdout
                break
        mlflow.log_text(gpu_info or "No GPU found.", "gpu-info.txt")

        # 设备 & dtype
        device = setup_device()
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Using device: {device}, dtype: {dtype}")

        # —— 5. 初始化 Tokenizer & 量化配置 —— #
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=True,
            padding_side="right",
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

        # —— 6. 量化加载模型 —— #
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=dtype,
            use_auth_token=True,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # —— 7. 准备数据集 —— #
        raw_ds = load_dataset("qa_from_commits_formatted.json")
        processor = LlamaQAPreprocessor(tokenizer, max_length=128)
        processed_ds = raw_ds.map(
            processor,
            batched=True,
            batch_size=4,
            remove_columns=["question", "context", "answer"],
            num_proc=4,
        )
        processed_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # —— 8. 训练参数 —— #
        training_args = TrainingArguments(
            output_dir="/mnt/object/data/llama3_qa_4bit",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,  # —↓ 更少的累积步数
            num_train_epochs=3,
            learning_rate=2e-5,
            bf16=True,
            fp16=False,
            gradient_checkpointing=True,
            optim="adamw_torch",
            logging_steps=100,
            save_steps=500,
            report_to="mlflow",
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            dataloader_drop_last=True,
        )

        mlflow.log_params({
            "model": model_name,
            "batch_size": training_args.per_device_train_batch_size,
            "grad_accumulation": training_args.gradient_accumulation_steps,
            "learning_rate": training_args.learning_rate,
            "device": device,
            "quant": "4bit_nf4",
            "max_length": 128,
        })

        # —— 9. 开始训练 —— #
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_ds,
            tokenizer=tokenizer,
        )
        print("开始训练（4-bit 量化）…")
        trainer.train()

        # —— 10. 保存 & 上报模型 —— #
        trainer.save_model(training_args.output_dir)
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="commit_qa_llama3_model_4bit",
            task="text-generation",
        )
        print("训练完成，量化模型已保存。")
