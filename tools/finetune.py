import json
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
model_id = str(PROJECT_ROOT / "models" / "Qwen2.5-1.5B")
dataset_path = str(PROJECT_ROOT / "data" / "psyqa" / "psyqa_train.jsonl")
output_dir = str(PROJECT_ROOT / "models" / "Qwen2.5-1.5B-PsyQA-LoRA")

def prepare_dataset(file_path, num_samples=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if num_samples is not None and i >= num_samples:
                break
            d = json.loads(line)
            question = d.get('question', '')
            desc = d.get('description', '')
            user_msg = question
            if desc:
                user_msg += "\n" + desc
            answers = d.get('answers', [])
            if not answers:
                continue
            answer = answers[0].get('answer_text', '')
            if not answer:
                continue
            
            messages = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": answer}
            ]
            data.append({"messages": messages})
    return Dataset.from_list(data)

def main():
    print("Loading dataset...")
    dataset = prepare_dataset(dataset_path, num_samples=None)  # 取消截断，使用全部数据
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_data(example):
        example["text"] = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return example
    
    print("Formatting dataset...")
    dataset = dataset.map(format_data)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    
    print("Preparing LoRA...")
    lora_config = LoraConfig(
        r=32,               # 从 8 提升到 32，增加可训练参数容量
        lora_alpha=64,      # 相应提升到 64 保持稳定性
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Starting training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,      # 等效批次大小为 2*4=8
        num_train_epochs=3,                 # 进行完整的 3 个 Epoch 训练 (替代原来的 max_steps)
        warmup_steps=100,                   # 增加预热步数，保护训练稳定性
        learning_rate=1e-4,                 # 全量微调可以适当调低学习率
        fp16=True,
        logging_steps=50,                   # 调整日志输出频率
        save_strategy="epoch",              # 每一个 Epoch 保存一次检查点
        optim="adamw_torch",
        report_to="none"
    )
    
    from transformers import Trainer, DataCollatorForLanguageModeling
    
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_dataset,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    
    trainer.train()
    
    print("Saving model...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
