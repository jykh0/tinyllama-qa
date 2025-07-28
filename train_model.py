import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import torch
MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MODEL_DIR="models/TinyLlama-1.1B-Chat-v1.0"
LORA_OUTPUT_DIR="models/TinyLlama-1.1B-Chat-v1.0-lora"
DATA_PATH="trainingdata/QqA_Pairs.jsonl"
train_batch_size=1
torch_dtype=torch.float32
device_map=None
use_fp16=False
def load_qa_dataset(path):
    with open(path,encoding="utf-8") as f:
        data=[json.loads(line) for line in f if line.strip()]
    for d in data:
        d["text"]=f"### Instruction:\n{d['instruction']}\n\n### Response:\n{d['output']}"
    return Dataset.from_list(data)
dataset=load_qa_dataset(DATA_PATH)
tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME,cache_dir=MODEL_DIR)
model=AutoModelForCausalLM.from_pretrained(MODEL_NAME,cache_dir=MODEL_DIR,torch_dtype=torch_dtype,device_map=device_map)
lora_config=LoraConfig(r=4,lora_alpha=8,target_modules=["q_proj","v_proj"],lora_dropout=0.05,bias="none",task_type="CAUSAL_LM")
model=get_peft_model(model,lora_config)
def tokenize_function(example):
    return tokenizer(example["text"],truncation=True,max_length=256,padding="max_length")
tokenized_dataset=dataset.map(tokenize_function,batched=False)
data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
training_args=TrainingArguments(output_dir=LORA_OUTPUT_DIR,per_device_train_batch_size=train_batch_size,num_train_epochs=1,learning_rate=2e-4,fp16=use_fp16,save_strategy="epoch",logging_steps=5,report_to=[],save_total_limit=1)
trainer=Trainer(model=model,args=training_args,train_dataset=tokenized_dataset,data_collator=data_collator)
trainer.train()
model.save_pretrained(LORA_OUTPUT_DIR)
