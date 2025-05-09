#!/usr/bin/env python3
import os
import json
import subprocess
import mlflow
import mlflow.pytorch
import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset as HFDataset

# —— 1. 数据预处理 —— #
def preprocess_data(file_path: str) -> HFDataset:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    formatted = [{
        "input": f"Q: {item['question']} C: {item['context']}",
        "answer": item['answer']
    } for item in data]
    return HFDataset.from_dict({
        "input": [x["input"] for x in formatted],
        "answer": [x["answer"] for x in formatted]
    })

# —— 2. Tokenizer + labels 处理器 —— #
class CommitQAPreprocessor:
    def __init__(self, tokenizer, max_input_length=192, max_target_length=96):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, examples):
        inputs = self.tokenizer(
            examples["input"],
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length",
        )
        labels = self.tokenizer(
            examples["answer"],
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
        )["input_ids"]
        inputs["labels"] = labels
        return inputs

# —— 3. 设备选择 —— #
def setup_device():
    if torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.9)
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

if __name__ == "__main__":
    # —— MLflow 跟踪设置 —— #
    os.environ["MLFLOW_TRACKING_URI"] = "http://129.114.108.6:8000/"
    mlflow.set_experiment("Commit QA Training - Optimized")

    # —— 先结束任何遗留的 run —— #
    try:
        mlflow.end_run()
    except Exception:
        pass

    # —— 用系统指标开一个新 Run —— #
    with mlflow.start_run(log_system_metrics=True):
        # ——— 记录 GPU/CPU 信息为 artifact ——— #
        gpu_cmds = ["nvidia-smi", "rocm-smi -v"]
        gpu_info = None
        for cmd in gpu_cmds:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout
                break
        if gpu_info is None:
            gpu_info = "No GPU found."
        mlflow.log_text(gpu_info, "gpu-info.txt")

        # —— 模型 & Tokenizer 初始化 —— #
        model_name = "t5-small"
        tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        model = T5ForConditionalGeneration.from_pretrained(model_name)

        # —— 设备搬模型 —— #
        device = setup_device()
        model.to(device)
        print(f"Using device: {device}")

        # —— 加载 & 预处理数据 —— #
        raw_ds = preprocess_data("model_training/qa_from_commits_formatted.json")
        processed_ds = raw_ds.map(
            CommitQAPreprocessor(tokenizer),
            batched=True,
            batch_size=256,
            remove_columns=["input", "answer"],
            num_proc=4
        )
        processed_ds.set_format(type=None)

        # —— 训练参数 —— #
        training_args = Seq2SeqTrainingArguments(
            output_dir="/home/jovyan/work",    # ← 已去掉末尾空格
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            learning_rate=5e-5,
            warmup_steps=500,
            weight_decay=0.01,
            fp16=(device == "cuda"),
            gradient_checkpointing=True,
            max_grad_norm=1.0,
            optim="adafactor",
            lr_scheduler_type="cosine",
            save_strategy="steps",
            save_steps=1000,
            logging_steps=100,
            report_to="mlflow",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            push_to_hub=False,
        )

        # —— DataCollator —— #
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        # —— 记录超参数 —— #
        mlflow.log_params({
            "model": model_name,
            "batch_size": training_args.per_device_train_batch_size,
            "grad_accumulation": training_args.gradient_accumulation_steps,
            "device": device
        })

        # —— 启动训练 —— #
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=processed_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        print("开始训练…")
        trainer.train()

        # —— 保存 & 上传模型 —— #
        trainer.save_model(training_args.output_dir)

        # 1) 使用 PyTorch flavor 上传模型
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="commit_qa_model_pytorch"
        )

        # 2) 保存并上传 tokenizer
        tokenizer_dir = os.path.join(training_args.output_dir, "tokenizer")
        tokenizer.save_pretrained(tokenizer_dir)
        mlflow.log_artifacts(
            local_dir=tokenizer_dir,
            artifact_path="commit_qa_tokenizer"
        )

        print("训练完成，模型和 tokenizer 已保存并上传到 MLflow。")
