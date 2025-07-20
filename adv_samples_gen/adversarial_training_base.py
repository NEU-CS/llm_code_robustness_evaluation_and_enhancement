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
tokenizer = AutoTokenizer.from_pretrained(basemodel)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    inputs = []
    for i in range(len(examples['code_str_generate'])):
        code = examples['code_str_generate'][i]
        truth = examples['Adversarial truth'][i]
        prompt = f"""Question:
{code}
Answer:
{truth}"""
        inputs.append(prompt)
    
    assert len(inputs) > 0
    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=1024,
    )
    
    model_inputs["labels"] = model_inputs["input_ids"].copy()
    
    for i in range(len(inputs)):
        question_part = f"""Question:
{examples['code_str_generate'][i]}
Answer:
"""
        question_tokens = tokenizer(question_part, truncation=True, max_length=1024)["input_ids"]
        question_length = len(question_tokens)
        model_inputs["labels"][i][:question_length] = [-100] * question_length
    
    return model_inputs

tokenized_data = data.map(preprocess_function,batched=True,remove_columns=['Adversarial truth','code_str_generate'])
print(tokenized_data)

from transformers import AutoModelForCausalLM,TrainingArguments, Trainer
from peft import (
    LoraConfig,
    get_peft_model,
    PeftType,
    TaskType
)

peft_type = PeftType.LORA
config = LoraConfig(
    r=16,
    lora_alpha=32,
    inference_mode=False,
    lora_dropout=0.1,
    task_type=TaskType.CAUSAL_LM,
    target_modules=[
        "q_proj",
        "v_proj"
    ],
)

model = AutoModelForCausalLM.from_pretrained(basemodel, torch_dtype=torch.float16)
model = get_peft_model(model, config)
model.print_trainable_parameters()

num_epochs = 2
training_args = TrainingArguments(
    output_dir=f"Qwen2.5-Coder-{args.model_size}-Base-LoRA",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    weight_decay=0.01,
    num_train_epochs=num_epochs,
    warmup_ratio=0.1,
    fp16=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    save_total_limit=1,
    dataloader_drop_last=True,
    load_best_model_at_end=True,
    report_to=None,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors="pt"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    eval_dataset=tokenized_data['test']
)

trainer.train()
