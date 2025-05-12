#!/usr/bin/env python3
import os, json, subprocess, torch, mlflow
from datasets import Dataset as HFDataset, Features, Sequence, Value
from transformers import (
    LlamaForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType

# —————————————————————————————————————————————————————————————
# 1. Environment
# —————————————————————————————————————————————————————————————
os.environ["MLFLOW_TRACKING_URI"]      = "http://129.114.108.60:8000"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64,garbage_collection_threshold:0.6"

# —————————————————————————————————————————————————————————————
# 2. Custom Trainer for LoRA: ensure labels→loss mapping
# —————————————————————————————————————————————————————————————
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

# —————————————————————————————————————————————————————————————
# 3. Load & preprocess dataset
# —————————————————————————————————————————————————————————————
def load_dataset(path: str) -> HFDataset:
    data = json.load(open(path, "r", encoding="utf-8"))
    return HFDataset.from_dict({
        "question": [d["question"] for d in data],
        "context":  [d["context"]  for d in data],
        "answer":   [d["answer"]   for d in data],
    })

class Preprocessor:
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def __call__(self, examples):
        ids, masks, labs = [], [], []
        for q, c, a in zip(examples["question"], examples["context"], examples["answer"]):
            prompt_ids   = self.tokenizer(f"问题：{q}\n上下文：{c}\n回答：",
                                          add_special_tokens=False
                                         )["input_ids"]
            response_ids = self.tokenizer(a + self.tokenizer.eos_token,
                                          add_special_tokens=False
                                         )["input_ids"]
            seq = (prompt_ids + response_ids)[: self.max_length]
            lbl = ([-100]*len(prompt_ids) + response_ids)[: self.max_length]
            pad = self.max_length - len(seq)
            seq += [self.tokenizer.pad_token_id]*pad
            lbl += [-100]*pad
            mask = [1]*(self.max_length-pad) + [0]*pad
            ids.append(seq); masks.append(mask); labs.append(lbl)
        return {"input_ids": ids, "attention_mask": masks, "labels": labs}

# —————————————————————————————————————————————————————————————
# 4. Main training flow
# —————————————————————————————————————————————————————————————
def main():
    mlflow.set_experiment("Commit QA Training")
    with mlflow.start_run():
        # Log GPU info
        for cmd in ["nvidia-smi","rocm-smi -v"]:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if r.returncode==0:
                mlflow.log_text(r.stdout, "gpu-info.txt")
                break

        device = "cuda" if torch.cuda.is_available() \
                 else "mps"  if torch.backends.mps.is_available() \
                 else "cpu"
        dtype  = torch.bfloat16 if device=="cuda" else torch.float32
        print(f"Device={device}, dtype={dtype}")

        # Quantized tokenizer & model
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        tokenizer  = AutoTokenizer.from_pretrained(model_name, use_auth_token=True, padding_side="right")
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            torch_dtype=dtype,
            use_auth_token=True,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # LoRA adapter
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

        # Dataset + mapping
        ds = load_dataset("qa_from_commits_formatted.json")
        schema = Features({
            "input_ids":      Sequence(Value("int64"), length=128),
            "attention_mask": Sequence(Value("int64"), length=128),
            "labels":         Sequence(Value("int64"), length=128),
        })
        proc = Preprocessor(tokenizer, max_length=128)
        ds   = ds.map(
            proc,
            batched=True,
            batch_size=4,
            remove_columns=["question","context","answer"],
            features=schema,
            num_proc=1,
        )
        ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

        # TrainingArguments
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
            "model":      model_name,
            "max_length": 128,
            "batch_size": args.per_device_train_batch_size,
            "accum_steps":args.gradient_accumulation_steps,
            "lr":         args.learning_rate,
            "lora_r":     lora_cfg.r,
        })

        # Train
        trainer = MyTrainer(model=model, args=args, train_dataset=ds, tokenizer=tokenizer)
        trainer.train()

        # Save
        model.save_pretrained(args.output_dir)
        print(f"Saved to {args.output_dir}")

if __name__=="__main__":
    main()
