import re


def split_on_labels(input_text, labels):
    # Check if input is actualy text
    assert type(input_text) == str, f"Input text isn't a string: {input_text}"
    
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
    
    # Check if input is actualy text
    if isinstance(text, list):
        text = text[0]
    assert type(text) == str, f"Input text isn't a string: {text}"

    # split text
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

def split_coferents(ent):
    '''
    This function splits a entitiy with a coferent mention into two entities for each entity form.
    '''
    if isinstance(ent['text'], tuple): # Check if the entity has coferent mentions
        return tuple([{"label":ent["label"], "text":ent['text'][i]} for i in range(len(ent['text']))])
    else:
        return (ent,)

def map_coferents(group):
    '''
    This function maps all forms of a coferent mentions to all it's other forms for all coferent mentions in a group of relationships.
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

def ner_metric(predictions: list[str], references: list[str], ner_labels: list[str], re_labels: list[str], coferent_matching: str ="relaxed") -> dict:
    '''
    Calculates the precision, recall and f1-score for document named entity recognition. 
    input:
        predictions: 
            List of decoded outputs of the model
        references: 
            List of decoded gold data
        coferent_matching: 
            Wheter to use the coferent mentions to match named entities. can be either "relaxed", "strict" or "no".
            "relaxed" Meaning that all coferent mentions might be used to match a predicted named entity to a reference entity
            "strict"  Meaning the model needs to have all coferent mentions correct to count as a match. (including the sequence)
            "no"      Meaning that coferent mentions are ignored, and only the first mentions are used.
        re_labels
            Which relation extraxtion labels are used.
    output
        a dictionary with the key value pairs of metric_name : metric value
    '''
    if coferent_matching not in ["relaxed", "strict", "no"]: 
        raise ValueError(f"'{coferent_matching}' is not a valid value for coferent_matching, Please choose one of ['relaxed', 'strict', 'no'].")

    if coferent_matching == "no":
        keep_coferents = False
    else:
        keep_coferents = True
    
    tp = 0 # True positive count
    fp = 0 # False positive count
    fn = 0 # False negative count
    unstructured_text_count = 0
    
    for pred_text, ref_text in zip(predictions, references):
        # Define groups
        try:
            pred_group = get_group(extract_relation_triples(pred_text, ner_labels, re_labels, keep_coferents))
        except ValueError:
            # Should be a logging statement here
            continue # Skip this row entirely
        
        ref_group = get_group(extract_relation_triples(ref_text, ner_labels, re_labels, keep_coferents))

        if coferent_matching == "relaxed":
            # Create mapping from a coferent mentions to all coferent mentions
            mapping_coferent = map_coferents(ref_group)
            
            # Split entities in the reference group
            ref_group = [split_coferents(ent) for ent in ref_group] # Split coferents into multiple entities 
            ref_group = [item for sublist in ref_group for item in (sublist if isinstance(sublist, tuple) else [sublist])] # Flatten list
        
            # print(f"pred_group: {pred_group}\n\nref_group: {ref_group}\n") #DEBUG
            # print(f"mapping: {mapping_coferent}\n\n\n") # DEBUG
        
        checked_coferent_pred = []
        for ent in pred_group:
            # print(f"entity: {ent}") # DEBUG
            # print(f"ref_group: {ref_group} ") # DEBUG

            if coferent_matching == "relaxed":
                # Split coferent entity
                ent_forms = split_coferents(ent)
            else:
                ent_forms = [ent]

            # print(f"\nStarting entity checking, ent_forms: {ent_forms}\n") # DEBUG
            for ent_form in ent_forms:
                # print(f"Checking if {ent_form} in {ref_group}") # DEBUG
                if ent_form in ref_group: # True positive
                    tp=tp+1
                    # print(f"True! \n") # DEBUG
                    
                    # Remove all instances of the coferent mentions
                    if coferent_matching == "relaxed": 
                        [ref_group.remove(i) for i in mapping_coferent[frozenset(ent_form.items())]] # Remove all coferent mentions from the reference group
                        # print(f"Removing coferent mentions from reference group: {[i for i in mapping_coferent[frozenset(ent_form.items())]]}\n") #DEBUG
                        checked_coferent_pred.extend([i for i in mapping_coferent[frozenset(ent_form.items())]]) # Remember which coferent mentions have been checked
                    else: 
                        ref_group.remove(ent) 
                    break # A match was found so we move on to the next entity
                    
                elif ent_form not in ref_group and ent_form not in checked_coferent_pred: # False positive
                    fp=fp+1
                    # print(f"False! \n") #DEBUG
                    break # A mismatch was found so we move on to the next entity

        # print(f"Counting false negatives. Current ref group: length:{len(ref_group)} | {ref_group}\n") # DEBUG
        # [ref_group.remove(i) for i in checked_coferent_pred if i in ref_group]
        # print(f"ref group after removeal of checked coferents: length:{len(ref_group)} | {ref_group}\n") # DEBUG

        # if coferent_matching == "relaxed": # WORK NEEDED. RELAXED MATCHING BASED ON COFERENT MENTIONS DOES NOT WORK YET!!!
            # print(f"checked_coferent_pred: {checked_coferent_pred}") #DEBUG
            # Remove all checked entities before counting false negatives
            
        fn=fn+len(ref_group) # False negative 
        # print(f"TP: {tp}, FP: {fp}, FN: {fn} \n\n\n") #DEBUG
    
    # Calculate metrics
    if (tp+fp) == 0: precision=0.0
    else: precision = tp/(tp+fp)
    
    if (tp+fn) == 0: recall=0.0
    else: recall = tp/(tp+fn)
    
    if (precision+recall) == 0: f1=0.0
    else: f1 = 2 * ((precision*recall)/(precision+recall))
    
    return {'ner_precision':precision, 'ner_recall':recall, 'ner_f1':f1}

def match_re_relaxed(predicted_triple, references):

    pred_head_mentions = predicted_triple["head_ent"]["text"]
    pred_tail_mentions = predicted_triple["tail_ent"]["text"]

    for head_mention in pred_head_mentions:
        for reference_triple in references:
            if head_mention in reference_triple["head_ent"]["text"] and reference_triple["head_ent"]["label"]==predicted_triple["head_ent"]["label"]: # Head entity match
                for tail_mention in pred_tail_mentions:
                    if tail_mention in reference_triple["tail_ent"]["text"] and reference_triple["tail_ent"]["label"]==predicted_triple["tail_ent"]["label"]: # Tail entity match
                        return (True, reference_triple)
    return (False, None)

def match_re_strict(predicted_triple, references):

    pred_head_mentions = predicted_triple["head_ent"]["text"]
    pred_tail_mentions = predicted_triple["tail_ent"]["text"]

    for reference_triple in references:
        if set(pred_head_mentions) == set(reference_triple["head_ent"]["text"]) and reference_triple["head_ent"]["label"]==predicted_triple["head_ent"]["label"]: # Head entity match
            for tail_mention in pred_tail_mentions:
                    if set(pred_tail_mentions) == set(reference_triple["tail_ent"]["text"]) and reference_triple["tail_ent"]["label"]==predicted_triple["tail_ent"]["label"]: # Tail entity match
                        return (True, reference_triple)
    return (False, None)

def re_metric(predictions: list[str], references: list[str], ner_labels: list[str], re_labels: list[str], coferent_matching: str ="relaxed"):
    
    tp = 0 # True positive count
    fp = 0 # False positive count
    fn = 0 # False negative count
    
    unstructured_text_count = 0

    if coferent_matching == "no":
        keep_coreforents = False
    else:
        keep_coreforents = True
    
    # Transform structed text into relationship triples
    for pred_text, ref_text in zip(predictions, references):
        try:
            predicted_triples = extract_relation_triples(pred_text, ner_labels, re_labels, keep_coreforents=keep_coreforents)
        except ValueError: # Text is unstructured
            unstructured_text_count += 1
            continue
    
        references = extract_relation_triples(ref_text, ner_labels, re_labels, keep_coreforents=keep_coreforents)

        # Determine matches between predicted and reference triples
        for predicted_triple in predicted_triples:
            # print(f"Checking if {predicted_triple}\nin") # DEBUG
            # print(f"references{references}\n") # DEBUG
            if coferent_matching == "relaxed":
                is_match, matched_ref_triple = match_re_relaxed(predicted_triple, references)
                if is_match: 
                    tp = tp + 1
                    # print(f"True!") # DEBUG
                    references.remove(matched_ref_triple)
                else:
                    fp = fp + 1
                    # print(f"False!") # DEBUG
                    
            elif coferent_matching == "strict" or "no":
                is_match, matched_ref_triple = match_re_strict(predicted_triple, references)
                if is_match: 
                    tp = tp + 1
                    # print(f"True!") # DEBUG
                    references.remove(matched_ref_triple)
                else:
                    fp = fp + 1
                    # print(f"False!") # DEBUG
        
                
        # False negative
        # print(f"Counting false negatives: {len(references)} from: {references} \n") #DEBUG
        fn+=len(references)
        # print(f"Current counts: tp:{tp} | fp:{fp} | fn:{fn} \n\n") # DEBUG

    # Calculate metrics
    if (tp+fp) == 0: precision=0.0
    else: precision = tp/(tp+fp)

    if (tp+fn) == 0: recall=0.0
    else: recall = tp/(tp+fn)

    if (precision+recall) == 0: f1=0.0
    else: f1 = 2 * ((precision*recall)/(precision+recall))

    unstructured_text = unstructured_text_count/len(predictions)

    return {'re_precision':precision, 're_recall':recall, 're_f1':f1, 'unstructured':unstructured_text}