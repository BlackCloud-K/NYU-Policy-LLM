import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token  # 重要：防止 padding 报错

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
)
model = get_peft_model(model, lora_config)

# 2. 加载你的指令式微调数据集（json 或 hf 数据集格式）
dataset = load_dataset("json", data_files="nyu_full_instruction_dataset.json")["train"]

def format_example(example):
    prompt = f"{example['instruction']}\n\n{example['input']}\n\nAnswer:"
    response = example["output"]
    return {"input_ids": tokenizer(prompt + response, truncation=True, padding="max_length", max_length=512)["input_ids"]}

tokenized_dataset = dataset.map(format_example)

# 3. 设置训练参数
training_args = TrainingArguments(
    output_dir="./lora-llama3-checkpoint",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# 4. 使用 Trainer 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

model.save_pretrained("./lora-mistral-adapter")
tokenizer.save_pretrained("./lora-mistral-adapter")
