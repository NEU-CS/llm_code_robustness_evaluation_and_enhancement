import os
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

path = "dataset/"
data = load_dataset("json",data_files={"train":os.path.join(path,"train.jsonl"),"test":os.path.join(path,"valid.jsonl")})


basemodel = "/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenzier = AutoTokenizer.from_pretrained(basemodel)
tokenzier.pad_token = tokenzier.eos_token


def preprocess_function(examples):
    inputs = []
    for i in range(len(examples['lang'])):
        code = examples['Adversarial Code'][i]
        order = examples['prompt'][i]
        x = ""
        lang = examples['lang'][i]
        if lang == "cpp":
            x = "5. Do not generate a main function, as I have my own main function available."
        elif lang == "java":
            x = "5. Do not modify class \"Solution\" as a public class."
        elif lang == "python":
            x = "5. Mind indent in python code."
        elif lang == "javascript":
            x = "5. Do not generate \"console.log\" statement, do not use \"require\" to import package."

        prompt = rf"""
Question:
This is a code generation task. Please help me write the code. The programming language for the code is {lang}. In the code, I have already provided a portion of it, and the remaining part needs to be completed by you. The placeholder 'begin to write code' is where you begin to complete the code.
The prompt for the code is: {order}
The code content is:
-----------------------------
{code}
-----------------------------

Requirements:
1. I only need the function and related package import, don't generate any other imformations such as examples usage or test cases.
2. Follow the specified format strictly below.
3. Do not change the function name.
4. The original code content must be fully included in the complete code you generate.
{x}

Format:
```{lang}
Complete code (including all the content of the code I provided and the code you generated)
```

Answer:
```{lang}
{examples['Adversarial truth'][i]}
```
'''
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

tokenzied_data = data.map(preprocess_function,batched=True,remove_columns=['Adversarial truth','Adversarial Code','lang'])

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
        lora_alpha=8,
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

num_epochs = 3
training_args = TrainingArguments(
    output_dir="Qwen2.5-Coder-1.5B-Instruct-LoRA",
    save_strategy = "epoch",
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size = 2,
    per_device_eval_batch_size = 1,
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