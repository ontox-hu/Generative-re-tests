from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch
from dataclasses import field
from typing import Optional
from datasets import load_dataset
from peft import LoraConfig
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
)
from peft.tuners.lora import LoraLayer
from trl import SFTTrainer
from os.path import abspath
import os
from pathlib import Path
from wasabi import msg
import logging

ex = Experiment()
ex.add_config('config/config.yaml')
ex.observers.append(FileStorageObserver('sacred_runs'))

@ex.capture
def create_lora_config(lora):
    """
    Creates a Lora config
    """
    peft_config = LoraConfig(
        lora_alpha=lora['alpha'],
        lora_dropout=lora['dropout'],
        r=lora['r'],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora['targets']
        )

    return peft_config

@ex.capture
def create_bitsandbytes_config(bnb_4bit, use_4bit, use_nested_quant):
    """
    Creates a bitsandbytes config
    """
    compute_dtype = getattr(torch, bnb_4bit['compute_dtype'])
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit['quant_type'],
        bnb_4bit_compute_dtype=bnb_4bit['compute_dtype'],
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    return bnb_config

def custom_load_dataset(dataset_vars):

    # Get the dataset file paths
    dataset_dir_path = Path(abspath("")).joinpath(dataset_vars['dir'])
    file_paths = dataset_dir_path.glob(f"*.{type}")

    # sacred observer | Log the dataset
    for path in file_paths:
        ex.open_resource(path.__str__())

    # Load dataset
    dataset = load_dataset(
        dataset_vars['type'], 
        data_dir=dataset_dir_path.__str__(),
        column_names=dataset_vars['column_names'],
        split=dataset_vars['split']
        )

    return dataset

@ex.automain
def main(
    bf16,
    # bnb_4bit_compute_dtype,
    # bnb_4bit_quant_type,
    dataset_vars,
    fp16,
    gradient_accumulation_steps,
    gradient_checkpointing,
    group_by_length,
    learning_rate,
    local_rank,
    logging_steps,
    # lora_alpha,
    # lora_dropout,
    # lora_r,
    lr_scheduler_type,
    max_grad_norm,
    max_seq_length,
    max_steps,
    model_name,
    num_train_epochs,
    optim,
    packing,
    per_device_eval_batch_size,
    per_device_train_batch_size,
    pytorch_cuda_alloc_conf_list,
    save_steps,
    # use_4bit,
    # use_nested_quant,
    warmup_ratio,
    weight_decay  
):

    # logging
    transformers.utils.logging.disable_progress_bar()
    
    # Setting Pytorch cuda allocation config
    for i in pytorch_cuda_alloc_conf_list: # WORK NEEDED doesn't this overwrite eachother?
        os.environ["PYTORCH_CUDA_ALLOC_CONF"]=i

    # Setting up
    peft_config = create_lora_config()
    bnb_config = create_bitsandbytes_config()
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
    )

    # Loading dataset
    with msg.loading(f"Loading dataset {dataset_vars['dir']}"):
        dataset = custom_load_dataset(dataset_vars)

        
    msg.good("Loaded dataset {dataset_vars['dir']}")

    # Load tokenizer
    with msg.loading(f"Initializing tokenizer"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    msg.good("Initialized Tokenizer")
    
    # Load model
    with msg.loading(f"Loading model {model_name}"):
        device_map = {"": 0} # FIND OUT WHAT THIS DOES
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
        )
    msg.good(f"Loaded model {model_name}")

    # Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field=dataset_vars[column_names][0], # could cause error
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train
    transformers.utils.logging.enable_progress_bar()
    trainer.train()


