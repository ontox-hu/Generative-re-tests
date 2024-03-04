import os
import logging
from os.path import abspath
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver
from dataclasses import field
from typing import Optional
from datasets import load_dataset
import transformers
import evaluate
import torch
import numpy as np
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from wasabi import msg

ex = Experiment()
ex.add_config('config/config_testing.yaml')
ex.observers.append(FileStorageObserver('sacred_runs'))

def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, rouge_types=['rouge1', 'rouge2'], references=decoded_labels, use_stemmer=False)
    result = {k: round(v * 100, 4) for k, v in result.items()} # rounds all metric values to 4 numvers behind the comma and make them percentages
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens) # mean length of the generated sequences
    return result

def preprocess_function(examples):
    '''
    This function takes a dataset of input and target sequences.
    meant to be used with the dataset.map() function
    '''

    text_column = dataset_vars['column_names'][0]
    rel_column = dataset_vars['column_names'][1]

    # Split input and target
    inputs, targets = [], []
    for i in range(len(examples[text_column])):
        if examples[text_column][i] and examples[rel_column][i]: # remove pairs where one is None
            inputs.append(examples[text_column][i])
            targets.append(examples[rel_column][i])

    # Tokenize the input
    model_inputs = tokenizer(
        inputs, 
        max_length=max_seq_length, 
        padding=padding, 
        truncation=truncation, 
        return_tensors='pt'
    )

    # Tokenize the target sequence
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=max_seq_length, 
            padding=padding, 
            truncation=truncation,  
            return_tensors='pt'
        )

    # Replace pad tokens with -100 so they don't contribute too the loss
    if ignore_pad_token_for_loss:
        labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

    # Add tokenized target text to output
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs
        

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
        column_names=dataset_vars['column_names']
        )

    return dataset

@ex.automain
def main(
    bf16,
    # bnb_4bit_compute_dtype,
    # bnb_4bit_quant_type,
    dataset_vars,
    output_dir,
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
    weight_decay,
    do_eval,
    evaluation_strategy,
    eval_steps
):

    # logging
    transformers.utils.logging.disable_progress_bar()

    
    
    # Setting Pytorch cuda allocation config
    # for i in pytorch_cuda_alloc_conf_list: # WORK NEEDED doesn't this overwrite eachother?
    #     os.environ["PYTORCH_CUDA_ALLOC_CONF"]=i

    # Setting up 
    # Note to self: if all of these parameters are defined with the same name in the config i 
    # could replace this with Seq2SeqTrainingArguments() because of the variable injections of sacred.
    training_arguments = Seq2SeqTrainingArguments(
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
        predict_with_generate=True,
        save_total_limit=2,
        save_strategy='steps',
        load_best_model_at_end=True,
        do_eval=do_eval,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps
    )

    # Loading dataset
    with msg.loading(f"Loading dataset from:{dataset_vars['dir']}"):
        dataset = custom_load_dataset(dataset_vars)
        dataset_train = dataset['train'].select(range(1,501)) # remove first row that contains column names
        dataset_eval = dataset['validation'].select(range(1,501)) # remove first row that contains column names
    msg.good(f"Loaded dataset from:{dataset_vars['dir']}")

    # Load tokenizer
    with msg.loading(f"Initializing tokenizer"):
        tokenizer = T5Tokenizer.from_pretrained(model_name, trust_remote_code=True)
    msg.good("Initialized Tokenizer")
    
    # Load model
    with msg.loading(f"Loading model {model_name}"):
        device_map = {"": 0} # FIND OUT WHAT THIS DOES
        model = T5ForConditionalGeneration.from_pretrained(
            model_name,
            load_in_8bit=use_8bit,
            device_map=device_map,
        )
    msg.good(f"Loaded model {model_name}")

    # Apply preprocessing
    with msg.loading(f"Preprocessing dataset..."):
        # Preprocess training dataset
        train_dataset = dataset_train.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on train dataset"
        )

        # Preprocess evaluation dataset
        eval_dataset = dataset_eval.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on train dataset"
        )
    msg.good(f"Preprocessed dataset!")

    # Load metric
    metric = evaluate.load("rouge")

    # Create Seq2Seq data collator to overwrite the default datacollator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if fp16 else None,
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        args=training_arguments,
        packing=packing
    )

    # Train
    transformers.utils.logging.enable_progress_bar()
    trainer.train()
    trainer.save_model(output_dir="fine_tune_results/final_model")


