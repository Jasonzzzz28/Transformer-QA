# train_custom.py
import json
import mlflow
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 改为文本分类模型

# 根据实际数据结构修改的数据集类
class CodeAnalysisDataset(Dataset):
    def __init__(self, data_path, max_length=128):
        with open(data_path) as f:
            self.raw_data = json.load(f)
        
        # 示例特征提取（根据实际数据结构调整）
        self.samples = []
        for item in self.raw_data:
            # 提取代码特征
            code = item.get('code', '')  # 使用code字段作为输入
            docstring = item.get('docstring', '')
            
            # 组合特征文本
            text = f"[CLS] {docstring} [SEP] {code}"
            
            # 示例标签（根据实际任务定义）
            label = 0 if "error" in code.lower() else 1  # 假设的简单分类
            
            self.samples.append({
                "text": text,
                "label": label
            })

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        encoding = self.tokenizer(
            sample["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "label": torch.tensor(sample["label"], dtype=torch.long)
        }

# 更新后的配置参数
config = {
    "experiment_name": "code_analysis",
    "model_name": "microsoft/codebert-base",
    "num_epochs": 10,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "max_seq_length": 256,
    "num_labels": 2
}

# 初始化MLFlow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(config["experiment_name"])

class CodeClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            config["model_name"],
            num_labels=config["num_labels"]
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            outputs = model(input_ids, attention_mask, labels=labels)
            
            total_loss += outputs.loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

if __name__ == "__main__":
    # 加载数据
    data_path = os.getenv("CUSTOM_DATA_PATH")
    dataset = CodeAnalysisDataset(data_path, max_length=config["max_seq_length"])
    
    # 划分数据集
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CodeClassifier(config).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])

    # MLFlow记录
    with mlflow.start_run():
        mlflow.log_params(config)
        
        best_val_acc = 0
        patience = 3
        patience_counter = 0

        for epoch in range(config["num_epochs"]):
            start_time = time.time()
            
            train_loss, train_acc = train(model, train_loader, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, device)
            
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }, step=epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")
                mlflow.log_artifact("best_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            print(f"Epoch {epoch+1}/{config['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
                  f"Time: {time.time()-start_time:.2f}s")

        # 测试评估
        test_loss, test_acc = validate(model, test_loader, device)
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})
        
        # 保存完整模型
        torch.save(model, "final_model.pth")
        mlflow.log_artifact("final_model.pth")
        mlflow.pytorch.log_model(model, "model")
