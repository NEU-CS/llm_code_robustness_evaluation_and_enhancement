import os
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--model_size", default="0.5B", type=str)


args = parser.parse_args()
path = "dataset/"
data = load_dataset("json",data_files={"train":os.path.join(path,"train_base.jsonl"),"test":os.path.join(path,"valid_base.jsonl")})


basemodel = f"/data1/model/qwen/Qwen/Qwen2.5-Coder-{args.model_size}"
tokenzier = AutoTokenizer.from_pretrained(basemodel)
#tokenzier.pad_token = tokenzier.eos_token


def preprocess_function(examples):
    inputs = []
    for i in range(len(examples['code_str_generate'])):
        code = examples['code_str_generate'][i]
        #order = examples[f'prompt'][i]
        truth = examples['Adversarial truth'][i]
        prompt = f"""
Question:
{code}
Answer:
{truth}
"""
        inputs.append(prompt)
        
    
    assert len(inputs) > 0
    model_inputs = tokenzier(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    return model_inputs

tokenzied_data = data.map(preprocess_function,batched=True,remove_columns=['Adversarial truth','code_str_generate'])
print(tokenzied_data)
from transformers import AutoModelForCausalLM,TrainingArguments, Trainer
from peft import (
    LoraConfig,
    get_peft_model,
    PeftType,
    TaskType
)
import torch
peft_type = PeftType.LORA
config = LoraConfig(
        r=8,
        lora_alpha=16,
        inference_mode=False,
        lora_dropout=0.1,
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
        "q_proj",
        "v_proj",
    ],
    )

model = AutoModelForCausalLM.from_pretrained(basemodel)
model = get_peft_model(model, config)
model.print_trainable_parameters()

num_epochs = 2
training_args = TrainingArguments(
    output_dir=f"Qwen2.5-Coder-{args.model_size}-Base-LoRA",
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    learning_rate=5e-5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size = 2,
    gradient_accumulation_steps= 2 ,
    weight_decay=0.01,
    num_train_epochs=num_epochs,
    warmup_ratio=0.1,
    fp16=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=1,
    dataloader_drop_last = True,
    load_best_model_at_end = True,
)

from transformers import DataCollatorForLanguageModeling

class DebugCollator(DataCollatorForLanguageModeling):
    def __call__(self, features):
        batch = super().__call__(features)
        return batch


data_collator = DebugCollator(tokenizer=tokenzier,mlm=False,return_tensors="pt")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenzied_data["train"],
    tokenizer=tokenzier,
    data_collator=data_collator,
    eval_dataset = tokenzied_data['test']
)

trainer.train()