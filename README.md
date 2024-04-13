# Seq2rel
This repository uses the seq2rel method of the paper [A sequence-to-sequence approach for document-level relation extraction](http://arxiv.org/abs/2204.01098) to extract relationships between chemicals and adverse outcomes described in scientific literature. 

---
# installation:

Create environment: 
```
python -m venv venv
```

Install torch manually [according to the installation guide](https://pytorch.org/get-started/locally/)
```
pip install torch
```

Copy environment 
```
pip install -r requirements.txt
```
# Use
The repository holds the code to fine-tune a huggingface seq2seq model. You can fine-tune a model by using the following command:

```
python run.py
```

this will start the training loop and train according to the config file defined in `run.py`.

The code makes use of the [sacred](https://github.com/IDSIA/sacred) module. This is a module that automaticly saves information about each run, making the experiments more reproducible. 

this means `run.py` has all the features of a sacred experiment:
you can see some of it's functionallity:
```
python run.py --help
```

and you could print the config:
```
python run.py print_config
```

## Backup Script
This Bash script is designed to automate the backup process for directories containing checkpoints and sacred runs. It takes two arguments: run_time and sacred_run_number, and performs the following actions:

1. **Clear Cache**: Clears the cache directory (`~/.cache`) on an Ubuntu system to ensure that cached data does not interfere with the backup process.

2. **Copy Checkpoints**: Scans the directory `Generative-re-tests/results` for directories named according to the convention "checkpoint-****". Copies any found directories to the designated checkpoint directory (`data/generative_re_model_storage_azure/<sacred_run_number>/checkpoints`).

3. **Copy Latest Sacred Run**: Checks the directory `Generative-re-tests/sacred_runs` for the most recently created directory with a name matching sacred_run_number. If found, copies it to `data/generative_re_model_storage_azure/<sacred_run_number>`.

4. **Report**: Prints a report at the end of the script, indicating the number of iterations, directories copied, and the total time elapsed.

### Usage
```
./backup_script.sh <run_time> <sacred_run_number>
```
- <run_time>: The duration (in seconds) for which the script should run.
- <sacred_run_number>: The identifier for the sacred run to be backed up.

---
# Installed packages:
- torch
- transformers
- accelerate
- datasets
- evaluate
- rouge_score
- sentencepiece
- protobuf
- ipykernel
- wasabi
- sacred