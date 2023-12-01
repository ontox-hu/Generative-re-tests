from sacred import Experiment
import torch
from dataclasses import field
from typing import Optional
from datasets import load_dataset
from peft import LoraConfig
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


ex = Experiment()

@ex.config
def train_config():    
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """
    # Type  str  | The hugging face model to use
    model_name = "tiiuae/falcon-7b"
    
    # Type  str  | The hugging face dataset to use
    dataset_name = ""

    # Type  int  |
    local_rank = -1

    # Type  int  |
    per_device_train_batch_size = 1
    # Type  int  |
    per_device_eval_batch_size = 1
    # Type  int  |
    gradient_accumulation_steps = 8
    # Type float |
    learning_rate = 2e-4
    # Type float |
    max_grad_norm = 0.3
    # Type float |
    weight_decay = 0.001
    # Type  int  |
    max_seq_length = 512
    # Type  int  |
    num_train_epochs = 1
    # Type Bool  |
    packing = False
    # Type Bool  |
    gradient_checkpointing  = True
    # Type  str  |
    optim  = "paged_adamw_32bit"
    # Type  str  |
    lr_scheduler_type = "constant"
    # Type  int  |
    max_steps = 10000
    # Type float |
    warmup_ratio = 0.03
    # Type Bool  |
    group_by_length = True
    # Type  int  |
    save_steps = 10

    #######################################################################################################################
    # These parameters controll how much GPU memory fine-tuning needs by effecting the Lora, and 4-bit training strategies#
    #######################################################################################################################
    
    # pytorch cuda memory allocation config
    pytorch_cuda_alloc_conf_list = ["heuristic", "max_split_size_mb512"]
    
    ###########  For Lora config ########## 
    lora_alpha = 128
    lora_dropout = 0.1
    lora_r  = 64
    lora_targets=[
            "query_key_value",
            "dense",
            "dense_h_to_4h",
            "dense_4h_to_h",]
    
    ##########  For bits and bytes config ########## 
    use_4bit = True
    use_nested_quant = False
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    
    ##########  for 16 byte training (?) ########## 
    fp16 = False
    bf16 = False

    ### general ####
    # Log every X updates steps during training.
    logging_steps = 10 

@ex.capture
def create_lora_config(lora_alpha, lora_dropout, lora_r, lora_targets):
    """
    Creates a Lora config
    """
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets
        )

    return peft_config

@ex.capture
def create_bitsandbytes_config(bnb_4bit_compute_dtype, use_4bit, bnb_4bit_quant_type, use_nested_quant):
    """
    Creates a bitsandbytes config
    """
    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_nested_quant,
    )
    
    return bnb_config

@ex.automain
def main(
    bf16,
    # bnb_4bit_compute_dtype,
    # bnb_4bit_quant_type,
    dataset_name,
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
    
    # Setting Pytorch cuda allocation config
    for i in pytorch_cuda_alloc_conf_list: # WORK NEEDED doesn't this overwrite eachother?
        os.environ["PYTORCH_CUDA_ALLOC_CONF"]=i

    # Setting up
    peft_config = create_lora_config()
    bnb_config = create_bitsandbytes_config()
    training_arguments = TrainingArguments(
        output_dir="./results", # WORK NEEDED add to config
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
    

    # Load model
    device_map = {"": 0} # FIND OUT WHAT THIS DOES
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
    )

    # fine tune script comes here

