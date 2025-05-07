import json
import mlflow
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, Adafactor
import torch
from torch.utils.data import DataLoader
import os
from datasets import Dataset as HFDataset
import numpy as np

# 设置MLflow Tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "http://129.114.109.139:8000/"
mlflow.set_experiment("Commit QA Training - Optimized")

# 1. 数据预处理优化（预生成缓存）
def preprocess_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    formatted_data = [{"input": f"Q: {item['question']} C: {item['context']}", "answer": item['answer']} 
                     for item in data]
    
    # 转换为HuggingFace Dataset对象以使用缓存
    hf_dataset = HFDataset.from_dict({
        "input": [x["input"] for x in formatted_data],
        "answer": [x["answer"] for x in formatted_data]
    })
    
    return hf_dataset

# 2. 数据集类优化（启用预加载）
class OptimizedCommitQADataset:
    def __init__(self, tokenizer, max_input_length=192, max_target_length=96):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def preprocess(self, examples):
        inputs = self.tokenizer(
            examples["input"],
            max_length=self.max_input_length,
            truncation=True,
            padding="max_length"
        )
        targets = self.tokenizer(
            examples["answer"],
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length"
        )
        
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": targets["input_ids"]
        }

# 3. 优化数据加载
def prepare_dataloader(dataset, tokenizer, batch_size=16):
    processor = OptimizedCommitQADataset(tokenizer)
    processed_dataset = dataset.map(
        processor.preprocess,
        batched=True,
        batch_size=256,
        num_proc=4,
        remove_columns=["input", "answer"],
        load_from_cache_file=True
    )
    processed_dataset.set_format(type="torch")
    
    return DataLoader(
        processed_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

# 初始化模型和分词器
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 加载并预处理数据
dataset = preprocess_data("model_training/qa_from_commits_formatted.json")
train_dataloader = prepare_dataloader(dataset, tokenizer)

# 启动MLflow运行
with mlflow.start_run():
    # 4. 优化训练参数
    training_args = TrainingArguments(
        output_dir="./commit_qa_model",
        per_device_train_batch_size=16,  # 增大batch size
        gradient_accumulation_steps=2,   # 梯度累积
        num_train_epochs=3,
        learning_rate=3e-4,
        warmup_steps=100,
        weight_decay=0.01,
        fp16=True,  # 启用混合精度训练
        gradient_checkpointing=True,  # 梯度检查点节省显存
        dataloader_num_workers=4,     # 并行数据加载
        save_strategy="steps",
        save_steps=1000,  # 减少保存频率
        logging_steps=200,
        report_to="mlflow",
        optim="adafactor",  # 使用更高效的优化器
        lr_scheduler_type="cosine",  # 余弦学习率调度
    )

    # 5. 自定义优化器配置
    def get_optimizer(model, args):
        return Adafactor(
            model.parameters(),
            lr=args.learning_rate,
            scale_parameter=True,
            relative_step=False,
            warmup_init=False
        )

    # 6. 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,  # 使用预处理后的数据集
        data_collator=lambda data: {
            "input_ids": torch.stack([f["input_ids"] for f in data]),
            "attention_mask": torch.stack([f["attention_mask"] for f in data]),
            "labels": torch.stack([f["labels"] for f in data])
        },
        optimizers=(get_optimizer(model, training_args), None)
    )

    # 记录超参数
    mlflow.log_params({
        "model_name": model_name,
        "batch_size": 16,
        "grad_accumulation": 2,
        "fp16": True,
        "optimizer": "adafactor"
    })

    # 7. 训练过程监控
    print("开始训练...")
    trainer.train()
    
    # 8. 模型保存优化
    output_dir = "./commit_qa_model"
    trainer.save_model(output_dir)
    
    # 记录模型时排除不需要的组件
    mlflow.transformers.log_model(
        transformers_model={
            "model": model,
            "tokenizer": tokenizer,
        },
        artifact_path="commit_qa_model",
        task="text2text-generation",
        input_example={"input": "Q: What's the purpose of commit e1f379b? C: {'commit_hash': 'e1f379b...}"}
    )

    print("优化训练完成！模型已记录到MLflow。")
