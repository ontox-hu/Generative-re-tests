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
import re
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from wasabi import msg

ex = Experiment()
ex.add_config('config/config_T5-L_cdr.yaml')
ex.observers.append(FileStorageObserver('sacred_runs'))

def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        # preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        # labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

def split_on_labels(input_text, labels):
    # Escape labels to ensure special characters are treated as literals in regex
    escaped_labels = [re.escape(label) for label in labels]
    # Join the labels into a regex pattern with alternation to match any of them
    pattern = '|'.join(escaped_labels)
    # Use re.split() with the compiled pattern, keeping the delimiters in the result
    relation_segments = re.split(f'({pattern})', input_text)
    # Filter out empty strings that might result from splitting
    relation_segments = [segment for segment in relation_segments if segment]
    return relation_segments

def handle_coreforents(ent, keep):
    coreferents = tuple([coref.strip() for coref in ent[0].split(';')])
    if keep and len(coreferents) > 1:
        return (coreferents, ent[1]) 
    else:
        return (coreferents[0], ent[1])

def extract_relation_triples(text: str, ner_labels: list[str], re_labels: list[str], keep_coreforents: bool = False) -> list[dict]:
    '''
    This function extracts the relationship triples out of structerd text. 
    This function assumes that the NER labels are in this structure: @label@
    
    input:
    text: The structerd text as a string.
    re_labels: The relationship labels.

    returns:
    A list of dictionaries
    '''
    ##### Check if text is structered #####
    split_on_space_text = text.split(" ")
    
    # check if text ends with a relation label
    if split_on_space_text[-1] not in re_labels:
        raise ValueError(f"Text is unstructured: '{text}'\nText should end with a relationship label found in re_labels: {re_labels}.\n")
        
    # Check if text has atleast two entity labels and one relation label
    count_ner_labels = sum([split_on_space_text.count(label) for label in ner_labels])
    count_re_labels = sum([split_on_space_text.count(label) for label in re_labels])
    if count_ner_labels < 2 or count_re_labels < 1:
        raise ValueError(f"Text is unstructured: '{text}'\nText should have atleast 2 ner_labels: {ner_labels} and 1 re_label: {re_labels} to make a relationship.\n")
        
    # Check if text has the right amount of entity and relation labels
    if count_re_labels*2 != count_ner_labels:
        raise ValueError(f"Text is unstructured: '{text}'\nText should have 2 times the ner_labels: {ner_labels} then there are re_label: {re_labels}. currently: ner labels: {count_ner_labels} | re labels: {count_re_labels}\n")
    
    ##### Extracting relation triples #####
    # Split the input text into relation segments
    relation_segments = split_on_labels(text, re_labels)
    
    # Remove the last empty segment if it exists
    if not relation_segments[-1].strip():
        relation_segments = relation_segments[:-1]

    # Map relation label to entity text
    entity_texts = relation_segments[::2] # All uneven elements
    relation_labels = relation_segments[1::2] # All even elements
        
    # Initialize a list to hold the relation triples
    relations = []
    
    for entity_text, re_label in zip(entity_texts, relation_labels):
        # Split head and tail entities and their labels
        head_ent, tail_ent = [handle_coreforents(ent, keep_coreforents) for ent in re.findall(r'(.+?)\s@(\w+)@', entity_text)]
        # print(f"head_ent {head_ent} | tail_ent {tail_ent}") #DEBUG
        
        re_label = re_label.split('@')[1]
        relations.append({
            're_label':re_label,
            'head_ent': {'label':head_ent[1], 'text':head_ent[0]},
            'tail_ent': {'label':tail_ent[1], 'text':tail_ent[0]}
        })
    
    return relations

def get_group(relation_triples):
    group = []
    for rel in relation_triples:
        group.append(rel['head_ent'])
        group.append(rel['tail_ent'])
    return group

def map_coferents(group):
    '''
    This function splits a coferent mention of an entity into the two different mentions
    '''
    result = {}
    group = [split_coferents(ent) for ent in group] # Split coferents into two entities
    for ent in group:
        if isinstance(ent, tuple):
            for i in range(len(ent)):
                result[frozenset(ent[i].items())] = ent
        else:
            result[frozenset(ent.items())] = (ent,)

    return result

def map_coferents(group):
    result = {}
    group = [split_coferents(ent) for ent in group] # Split coferents into two entities
    for ent in group:
        if isinstance(ent, tuple):
            for i in range(len(ent)):
                result[frozenset(ent[i].items())] = ent
        else:
            result[frozenset(ent.items())] = (ent,)

    return result

def ner_metric(predictions: list[str], references: list[str], ner_labels: list[str], re_labels: list[str], coferent_matching: str ="relaxed") -> dict:
    '''
    Calculates the precision, recall and f1-score for document named entity recognition. 
    input:
        predictions: 
            List of decoded outputs of the model
        references: 
            List of decoded gold data
        coferent_matching: 
            Wheter to use the coferent mentions to match named entities. can be either "relaxed" or "strict".
            relaxed meaning that all coferent mentions might be used to match a predicted named entity to a reference entity
            strict meaning the model needs to have all coferent mentions correct to count as a match. (including the sequence)
        re_labels
            Which relation extraxtion labels are used.
    output
        a dictionary with the key value pairs of metric_name : metric value
    '''
    tp = 0 # True positive count
    fp = 0 # False positive count
    fn = 0 # False negative count

    unstructured_text_count = 0
    
    for pred_text, ref_text in zip(predictions, references):
        # Define groups
        try:
            pred_group = get_group(extract_relation_triples(pred_text, ner_labels, re_labels, True))
        except ValueError:
            continue
            
        ref_group = get_group(extract_relation_triples(ref_text, ner_labels, re_labels, True))
    
        if coferent_matching == "relaxed":
            # Create mapping from a coferent mentions to all coferent mentions
            mapping_coferent = map_coferents(ref_group)
    
            pred_group = [split_coferents(ent) for ent in pred_group] # Split coferents into multiple entities 
            # print(f"pred_group before flattening: {pred_group}") # DEBUG
            pred_group = [item for sublist in pred_group for item in (sublist if isinstance(sublist, tuple) else [sublist])] # Flatten list
            # print(f"pred_group after flattening: {pred_group}"+'\n') # DEBUG
    
            ref_group = [split_coferents(ent) for ent in ref_group] # Split coferents into multiple entities 
            ref_group = [item for sublist in ref_group for item in (sublist if isinstance(sublist, tuple) else [sublist])] # Flatten list
    
        # print(f"pred_group: {pred_group}\nref_group: {ref_group}") #DEBUG
        # print(f"mapping: {mapping_coferent}")
        checked_coferent_pred = []
        for ent in pred_group:
            # print(ent) # DEBUG
            # print(ref_group) # DEBUG
            # print() #DEBUG
            if ent in ref_group: # True positive
                tp += 1
                # Remove all instances of the coferent mentions
                if coferent_matching == "relaxed": 
                    [ref_group.remove(i) for i in mapping_coferent[frozenset(ent.items())]]
                    checked_coferent_pred.extend([i for i in mapping_coferent[frozenset(ent.items())]])
                else: ref_group.remove(ent) 
                continue
            elif ent not in ref_group and ent not in checked_coferent_pred: # False positive
                fp += 1
        fn += len(ref_group) # False negative 
        # print(f"TP: {tp}, FP: {fp}, FN: {fn}") #DEBUG
    
    # Calculate metrics
    if (tp+fp) == 0: precision=0.0
    else: precision = tp/(tp+fp)
    
    if (tp+fn) == 0: recall=0.0
    else: recall = tp/(tp+fn)
    
    if (precision+recall) == 0: f1=0.0
    else: f1 = 2 * ((precision*recall)/(precision+recall))
    
    return {'ner_precision':precision, 'ner_recall':recall, 'ner_f1':f1}

def re_metric(predictions: list[str], references: list[str], ner_labels: list[str], re_labels: list[str]):
    
    tp = 0 # True positive count
    fp = 0 # False positive count
    fn = 0 # False negative count
    
    unstructured_text_count = 0
    
    # Define groups
    for pred_text, ref_text in zip(predictions, references):
        try:
            predicted_triples = extract_relation_triples(pred_text, ner_labels, re_labels, True)
        except ValueError: # Text is unstructured
            unstructured_text_count += 1
            continue
    
        references = extract_relation_triples(ref_text, ner_labels, re_labels, True)
    
        for pred in predicted_triples:
            if pred in references: # True positive
                tp=+1
                references.remove(pred)
            else: # False positive
                fp+=1
                
        # False negative
        fn+=len(references)

    # Calculate metrics
    if (tp+fp) == 0: precision=0.0
    else: precision = tp/(tp+fp)

    if (tp+fn) == 0: recall=0.0
    else: recall = tp/(tp+fn)

    if (precision+recall) == 0: f1=0.0
    else: f1 = 2 * ((precision*recall)/(precision+recall))

    unstructured_text = unstructured_text_count/len(predictions)

    return {'re_precision':precision, 're_recall':recall, 're_f1':f1, 'unstructured':unstructured_text}

@ex.capture
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
    result.update(re_metric(predictions=decoded_preds, references=decoded_labels, ner_labels=['@CHEMICAL@', '@DISEASE@'], re_labels=['@CID@']))
    result.update(ner_metric(predictions=decoded_preds, references=decoded_labels, ner_labels=['@CHEMICAL@', '@DISEASE@'], re_labels=['@CID@']))
    result = {k: round(v * 100, 4) for k, v in result.items()} # rounds all metric values to 4 numvers behind the comma and make them percentages
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens) # mean length of the generated sequences
    return result

@ex.capture
def preprocess_function(examples, dataset_vars, max_seq_length, padding, truncation, ignore_pad_token_for_loss):
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
    labels = tokenizer(
        text_target=targets, 
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
    dataset_vars,
    output_dir,
    fp16,
    gradient_accumulation_steps,
    gradient_checkpointing,
    group_by_length,
    learning_rate,
    local_rank,
    logging_steps,
    lr_scheduler_type,
    max_grad_norm,
    max_seq_length,
    max_steps,
    model_name,
    num_train_epochs,
    optim,
    per_device_eval_batch_size,
    per_device_train_batch_size,
    pytorch_cuda_alloc_conf_list,
    save_steps,
    use_8bit,
    warmup_ratio,
    weight_decay,
    do_eval,
    evaluation_strategy,
    eval_steps,
    ignore_pad_token_for_loss
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
        eval_steps=eval_steps,
        remove_unused_columns=True,
        generation_max_length=128
    )

    # Loading dataset
    with msg.loading(f"Loading dataset from:{dataset_vars['dir']}"):
        dataset = custom_load_dataset(dataset_vars)
        dataset_train = dataset['train'].select(range(1,501)) # remove first row that contains column names
        dataset_eval = dataset['validation'].select(range(1,501)) # remove first row that contains column names
    msg.good(f"Loaded dataset from:{dataset_vars['dir']}")

    # Load tokenizer
    with msg.loading(f"Initializing tokenizer"):
        global tokenizer #Otherwise the tokenizer won'te be acessible from within ohter functions
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, legacy=False, model_max_length=max_seq_length)
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

    ### Apply preprocessing
    with msg.loading(f"Preprocessing dataset..."):
        # Preprocess training dataset
        dataset_train = dataset_train.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on train dataset"
        )

        # Preprocess evaluation dataset
        dataset_eval = dataset_eval.map(
            preprocess_function,
            batched=True,
            desc="Running tokenizer on train dataset"
        )
    msg.good(f"Preprocessed dataset!")

    # Load metric
    global metric # Otherwise the metric object won't be accessible from within compute_metric()
    metric = evaluate.load("rouge")

    # Create Seq2Seq data collator to overwrite the default datacollator
    label_pad_token_id = -100 if ignore_pad_token_for_loss else tokenizer.pad_token_id
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
        args=training_arguments
    )

    # Train
    transformers.utils.logging.enable_progress_bar()
    trainer.train()
    trainer.save_model(output_dir="fine_tune_results/final_model")


