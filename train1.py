import json
import os
import subprocess
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

import mlflow
import mlflow.pytorch

# 配置MLFlow
mlflow.set_experiment("transformer-qa")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://129.114.108.179:8000/"))

# 训练配置
config = {
    "model_name": "bert-base-uncased",
    "max_length": 384,
    "stride": 128,
    "batch_size": 8,
    "learning_rate": 3e-5,
    "num_epochs": 10,
    "patience": 3,
    "max_grad_norm": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# 加载数据集
class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=384, stride=128):
        with open(data_path) as f:
            self.examples = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        encoding = self.tokenizer(
            example["question"],
            example["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        start_positions = []
        end_positions = []

        answer = example["answer"]
        for i, offsets in enumerate(encoding["offset_mapping"]):
            start_char = example["context"].find(answer)
            end_char = start_char + len(answer)

            sequence_ids = encoding.sequence_ids(i)
            start_idx = 0
            while sequence_ids[start_idx] != 1:
                start_idx += 1
            end_idx = len(sequence_ids) - 1
            while sequence_ids[end_idx] != 1:
                end_idx -= 1

            if start_char < offsets[start_idx][0] or end_char > offsets[end_idx][1]:
                start_positions.append(0)
                end_positions.append(0)
            else:
                while start_idx < len(offsets) and offsets[start_idx][0] <= start_char:
                    start_idx += 1
                while end_idx >= 0 and offsets[end_idx][1] >= end_char:
                    end_idx -= 1
                start_positions.append(start_idx - 1)
                end_positions.append(end_idx + 1)

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "start_positions": torch.tensor(start_positions),
            "end_positions": torch.tensor(end_positions)
        }


# 初始化模型和tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
model = AutoModelForQuestionAnswering.from_pretrained(config["model_name"])
model.to(config["device"])

# 数据加载器
dataset = QADataset(
    "/path/to/qa_from_commits_formatted.json",  # 替换实际路径
    tokenizer,
    max_length=config["max_length"],
    stride=config["stride"]
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)


# 训练函数
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "start_positions": batch["start_positions"].to(device),
            "end_positions": batch["end_positions"].to(device)
        }

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


# 验证函数
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "start_positions": batch["start_positions"].to(device),
                "end_positions": batch["end_positions"].to(device)
            }

            outputs = model(**inputs)
            total_loss += outputs.loss.item()
    return total_loss / len(dataloader)


# 初始化MLFlow运行
try:
    mlflow.end_run()
except:
    pass

with mlflow.start_run(log_system_metrics=True):
    # 记录系统信息
    rocm_info = subprocess.run(["rocm-smi"], capture_output=True, text=True).stdout
    mlflow.log_text(rocm_info, "rocm-info.txt")

    # 记录超参数
    mlflow.log_params(config)

    # 初始化优化器
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    best_loss = float("inf")
    patience_counter = 0

    # 训练循环
    for epoch in range(config["num_epochs"]):
        start_time = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, config["device"])
        val_loss = validate(model, val_loader, config["device"])

        epoch_time = time.time() - start_time

        # 记录指标
        mlflow.log_metrics({
            "epoch_time": epoch_time,
            "train_loss": train_loss,
            "val_loss": val_loss
        }, step=epoch)

        # Early stopping逻辑
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            mlflow.pytorch.log_model(model, "best_model")
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # 记录最终模型
    mlflow.pytorch.log_model(model, "final_model")

print("Training completed!")