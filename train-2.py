import os
import json
import subprocess
import mlflow
import torch
from datasets import Dataset as HFDataset
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# —— 1. 加载原始 JSON 数据 —— #
def load_dataset(file_path: str) -> HFDataset:
    """
    读取一个包含 {question, context, answer} 列表的 JSON 文件，
    返回 HuggingFace Dataset。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return HFDataset.from_dict({
        "question": [item["question"] for item in raw],
        "context":  [item["context"]  for item in raw],
        "answer":   [item["answer"]   for item in raw],
    })

# —— 2. 定义 Llama 专用的 Preprocessor —— #
class LlamaQAPreprocessor:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 确保有 pad_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, examples):
        input_ids_batch, attention_mask_batch, labels_batch = [], [], []

        for q, c, a in zip(examples["question"], examples["context"], examples["answer"]):
            # 构造 prompt
            prompt = f"### 问题：{q}\n### 上下文：{c}\n### 回答："
            # 分别编码 prompt 与 answer（带 eos）
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(a + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]

            # 拼接并截断到 max_length
            ids = prompt_ids + response_ids
            if len(ids) > self.max_length:
                ids = ids[: self.max_length]

            # 构造 labels：prompt 部分全 -100，response 部分保留
            labels = [-100] * len(prompt_ids) + response_ids
            if len(labels) > self.max_length:
                labels = labels[: self.max_length]

            # attention mask
            attention_mask = [1] * len(ids)

            # padding 到固定长度
            pad_len = self.max_length - len(ids)
            if pad_len > 0:
                ids            += [self.tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels         += [-100] * pad_len

            input_ids_batch.append(ids)
            attention_mask_batch.append(attention_mask)
            labels_batch.append(labels)

        return {
            "input_ids":      input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels":         labels_batch,
        }

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
    os.environ["MLFLOW_TRACKING_URI"] = "http://129.114.108.56:8000/"
    mlflow.set_experiment("Commit QA Training - Llama3.1-Instruct")

    # 关闭任何遗留的 run
    try:
        mlflow.end_run()
    except Exception:
        pass

    with mlflow.start_run(log_system_metrics=True):
        # —— 记录 GPU/CPU 信息为 artifact —— #
        gpu_info = None
        for cmd in ["nvidia-smi", "rocm-smi -v"]:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                gpu_info = r.stdout
                break
        if gpu_info is None:
            gpu_info = "No GPU found."
        mlflow.log_text(gpu_info, "gpu-info.txt")

        # —— 模型 & Tokenizer 初始化 —— #
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_auth_token=True,
            padding_side="right"
        )
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            use_auth_token=True,
        )

        # —— 设备搬模型 —— #
        device = setup_device()
        model.to(device)
        print(f"Using device: {device}")

        # —— 加载 & 预处理数据 —— #
        raw_ds = load_dataset("model_training/qa_from_commits_formatted.json")
        processor = LlamaQAPreprocessor(tokenizer, max_length=1024)
        processed_ds = raw_ds.map(
            processor,
            batched=True,
            batch_size=8,
            remove_columns=["question", "context", "answer"],
            num_proc=4,
        )
        processed_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # —— 训练参数 —— #
        training_args = TrainingArguments(
            output_dir="/mnt/object/llama3_qa_model",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=3,
            learning_rate=2e-5,
            fp16=(device == "cuda"),
            logging_steps=100,
            save_steps=500,
            report_to="mlflow",
            optim="paged_adamw_8bit",
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            dataloader_drop_last=True,
            push_to_hub=False,
        )

        # —— 记录超参数 —— #
        mlflow.log_params({
            "model":                 model_name,
            "per_device_batch_size": training_args.per_device_train_batch_size,
            "grad_accumulation":     training_args.gradient_accumulation_steps,
            "learning_rate":         training_args.learning_rate,
            "device":                device,
        })

        # —— 启动训练 —— #
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_ds,
            tokenizer=tokenizer,
        )
        print("开始训练…")
        trainer.train()

        # —— 保存 & 上报模型 —— #
        trainer.save_model(training_args.output_dir)
        mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="commit_qa_llama3_model",
            task="text-generation",
        )
        print("训练完成，模型已保存。")

