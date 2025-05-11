#!/usr/bin/env py:contentReference[oaicite:3]{index=3}mport subprocess

import torch
import mlflow
from dat:contentReference[oaicite:4]{index=4}ForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft impor:contentReference[oaicite:5]{index=5}—— 1. 加载数据 —— #
def load_d:contentReference[oaicite:6]{index=6}e_pat:contentReference[oaicite:7]{index=7} raw = json.load(f)
   :contentReference[oaicite:8]{index=8}quest:contentReference[oaicite:9]{index=9}] for item in raw],
        "context":  [item["context"]  for item in raw],
        "answer":   [item["answer"]   for item in raw],
 :contentReference[oaicite:10]{index=10}lamaQAPreprocessor:
    de:contentReference[oaicite:11]{index=11}x_length=128):
        self.tokenizer = tokenizer
        se:contentReference[oaicite:12]{index=12}tokenizer.:contentReference[oaicite:13]{index=13}all__(self, examples):
        in_ids, attn_masks, labels = [], [], []
        for q, c, a in zip(examples["question"], examples["context"], examples["answer"]):
            prompt       = f"### 问题：{q}\n### 上下文：{c}\n### 回答："
            prompt_ids   = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = self.tokenizer(a + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
            ids    = (prompt_ids + response_ids)[: self.max_length]
            labs   = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]
            mask   = [1] * len(ids)
            padlen = self.max_length - len(ids)
            if padlen > 0:
                ids   += [self.tokenizer.pad_token_id] * padlen
                mask  += [0] * padlen
                labs  += [-100] * padlen
            in_ids.append(ids)
            attn_masks.append(mask)
            labels.append(labs)
        return {"input_ids": in_ids, "attention_mask": attn_masks, "labels": labels}

# —— 3. 设备 —— #
def setup_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        torch.mps.set_per_process_memory_fraction(0.9)
        return "mps"
    else:
        return "cpu"

if __name__ == "__main__":
    # 环境变量
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"
    os.environ["MLFLOW_TRACKING_URI"]  = "http://129.114.108.60:8000"

    # 实验
    mlflow.set_experiment("Commit QA Training - Llama3.1-8B-4bit-LoRA")
    with mlflow.start_run(log_system_metrics=True):
        # — 记录 GPU 信息 — #
        gpu_info = None
        for cmd in ["nvidia-smi", "rocm-smi -v"]:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.returncode == 0:
                gpu_info = r.stdout
                break
        mlflow.log_text(gpu_info or "No GPU found.", "gpu-info.txt")

        # — 设定 device & dtype — #
        device = setup_device()
        dtype  = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"Using device={device}, dtype={dtype}")

        # — 4. Tokenizer & 4-bit 量化配置 — #
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

        # — 5. 量化加载模型 — #
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=dtype,
            use_auth_token=True,
        )
        model.config.use_cache = False

        # — 6. 注入 LoRA Adapter — #
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
        )
        model = get_peft_model(model, peft_config)

        # —— 确认哪些参数是可训练的 —— #
        print(">>> Trainable parameters after LoRA injection:")
        model.print_trainable_parameters()

        # —— （再保险）手动冻结 / 解冻 —— #
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # —— 7. 准备数据集 —— #
        raw_ds    = load_dataset("qa_from_commits_formatted.json")
        processor = LlamaQAPreprocessor(tokenizer, max_length=128)
        ds        = raw_ds.map(
            processor,
            batched=True,
            batch_size=4,
            remove_columns=["question","context","answer"],
            num_proc=4,
        )
        ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

        # —— 8. 训练参数 —— #
        training_args = TrainingArguments(
            output_dir="/mnt/object/data/llama3_qa_4bit_lora",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
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
            dataloader_drop_last=True,
        )

        # —— 9. 一次性记录所有超参 —— #
        mlflow.log_params({
            "model": model_name,
            "quant": "4bit_nf4",
            "lora_r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "lora_dropout": peft_config.lora_dropout,
            "max_length": 128,
            "batch_size": training_args.per_device_train_batch_size,
            "accum_steps": training_args.gradient_accumulation_steps,
            "lr": training_args.learning_rate,
        })

        # —— 10. Trainer & 训练 —— #
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=ds,
            tokenizer=tokenizer,  # 虽有 deprecation warning，但不影响
        )
        print(">>> 开始训练（4-bit + LoRA）…")
        trainer.train()

        # —— 11. 保存 LoRA adapter —— #
        model.save_pretrained(training_args.output_dir)
        print(f">>> LoRA adapter 和量化模型已保存到 {training_args.output_dir}")
    print("=== 完成 ===")
