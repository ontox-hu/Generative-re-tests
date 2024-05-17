#!/bin/bash

start_time=$(date +%s) # Start time of the operation

# Check if the correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <model_name> <dataset_name>"
    exit 1
fi

# Assigning arguments to variables
model_name=$1
dataset_name=$2
current_date=$(date +"%d_%m_%y")

# Creating new directory
new_directory="data/generative_re_model_storage_azure/ds_runs/${model_name}_${dataset_name}_${current_date}"
mkdir -p "$new_directory"

# Moving files and directories
mv_files=$(find data/generative_re_model_storage_azure/latest_run/* -maxdepth 0 -type f -exec mv {} "$new_directory" \; -print | wc -l)
mv_dirs=$(find data/generative_re_model_storage_azure/latest_run/* -maxdepth 0 -type d -exec mv {} "$new_directory" \; -print | wc -l)

# Removing jupyter checkpoints if if exists
rm -rf data/generative_re_model_storage_azure/latest_run/.ipynb_checkpoints/

# Calculating total size
mv_gb=$(du -ch "$new_directory" | grep total | cut -f1)

# End time of the operation
end_time=$(date +%s)
total_time=$((end_time - start_time))
minutes=$((total_time / 60))
seconds=$((total_time % 60))

# Printing summary
echo "Moved $mv_files files and $mv_dirs directories, total size: $mv_gb, took $minutes minutes and $seconds seconds."



