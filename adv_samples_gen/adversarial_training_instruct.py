import os
from datasets import load_dataset
from transformers import AutoTokenizer
import torch

path = "dataset/"
data = load_dataset("json",data_files={"train":os.path.join(path,"train.jsonl"),"test":os.path.join(path,"valid.jsonl")})

basemodel = "/data1/model/qwen/Qwen/Qwen2.5-Coder-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(basemodel)  
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    inputs = []
    targets = []  
    
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

        prompt = f"""Question:
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

Answer:"""
        

        target = f"```{lang}\n{examples['Adversarial truth'][i]}\n```"
        full_text = prompt + target
        
        inputs.append(full_text)
        targets.append(target)
    
    assert len(inputs) > 0
    
    model_inputs = tokenizer(
        inputs,
        padding="max_length",
        truncation=True,
        max_length=2048, 
        return_tensors="pt"
    )

    target_inputs = tokenizer(
        targets,
        padding="max_length", 
        truncation=True,
        max_length=2048,
        return_tensors="pt"
    )
    
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    
    for i in range(len(inputs)):
        prompt_length = len(tokenizer(prompt, truncation=True, max_length=2048)["input_ids"])
        model_inputs["labels"][i][:prompt_length] = -100
    
    return model_inputs

tokenized_data = data.map(preprocess_function, batched=True, remove_columns=['Adversarial truth','Adversarial Code','lang','prompt'])

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from peft import (
    LoraConfig,
    get_peft_model,
    PeftType,
    TaskType
)
import torch

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

num_epochs = 3
training_args = TrainingArguments(
    output_dir="Qwen2.5-Coder-1.5B-Instruct-LoRA",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=5e-5, 
    per_device_train_batch_size=1, 
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
