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
from transformers import TimeSeriesTransformerModel  # 使用时间序列Transformer模型


# 新增自定义数据加载类
class CodeDataset(Dataset):
    def __init__(self, data_path, seq_length=24, pred_length=6):
        with open(data_path) as f:
            raw_data = json.load(f)

        # 示例数据预处理 - 根据实际数据结构修改
        self.samples = []
        for item in raw_data:
            # 假设数据包含时间序列特征和标签
            features = np.array(item["features"], dtype=np.float32)
            labels = np.array(item["labels"], dtype=np.float32)

            # 创建滑动窗口样本
            for i in range(len(features) - seq_length - pred_length):
                self.samples.append({
                    "features": features[i:i + seq_length],
                    "labels": labels[i + seq_length:i + seq_length + pred_length]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return (
            torch.tensor(sample["features"], dtype=torch.float32),
            torch.tensor(sample["labels"], dtype=torch.float32)
        )


# 修改后的配置参数
config = {
    "experiment_name": "time_series_forecast",
    "model_architecture": "TimeSeriesTransformer",
    "num_epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "seq_length": 24,  # 输入序列长度
    "pred_length": 6,  # 预测长度
    "d_model": 128,  # Transformer维度
    "nhead": 4,  # 注意力头数
    "num_layers": 3,  # Transformer层数
    "dim_feedforward": 512,  # FFN维度
    "dropout": 0.1
}

# 初始化MLFlow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(config["experiment_name"])


class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transformer = TimeSeriesTransformerModel(
            input_size=config["d_model"],
            num_time_series=1,  # 单变量时间序列
            encoder_layers=config["num_layers"],
            decoder_layers=config["num_layers"],
            d_model=config["d_model"],
            nhead=config["nhead"],
            dim_feedforward=config["dim_feedforward"],
            dropout=config["dropout"]
        )
        self.projection = nn.Linear(config["d_model"], 1)

    def forward(self, src, tgt):
        output = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=None,
            tgt_mask=None,
            memory_mask=None
        ).last_hidden_state
        return self.projection(output)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output, tgt[:, 1:])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output, tgt[:, 1:])
            total_loss += loss.item()
    return total_loss / len(dataloader)


if __name__ == "__main__":
    # 加载自定义数据
    data_path = os.getenv("CUSTOM_DATA_PATH")
    dataset = CodeDataset(
        data_path,
        seq_length=config["seq_length"],
        pred_length=config["pred_length"]
    )

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
    model = TimeSeriesTransformer(config).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # MLFlow记录开始
    with mlflow.start_run():
        # 记录参数
        mlflow.log_params(config)

        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(config["num_epochs"]):
            start_time = time.time()

            train_loss = train(model, train_loader, criterion, optimizer, device)
            val_loss = validate(model, val_loader, criterion, device)

            # 记录指标
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=epoch)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")
                mlflow.log_artifact("best_model.pth")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

            print(f"Epoch {epoch + 1}/{config['num_epochs']} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Time: {time.time() - start_time:.2f}s")

        # 最终测试评估
        test_loss = validate(model, test_loader, criterion, device)
        mlflow.log_metric("test_loss", test_loss)

        # 保存完整模型
        torch.save(model, "final_model.pth")
        mlflow.log_artifact("final_model.pth")
        mlflow.pytorch.log_model(model, "model")