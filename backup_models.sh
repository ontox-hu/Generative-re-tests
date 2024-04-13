#!/bin/bash

# Check if the number of arguments is correct
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 run_time sacred_run_number"
    exit 1
fi

# Assign arguments to variables
run_time="$1"
sacred_run_number="$2"
start_time=$(date +%s)

# Function to clear cache
clear_cache() {
    cache_dir="$HOME/.cache"

    # Check if the cache directory exists
    if [ -d "$cache_dir" ]; then
        # Remove all files and directories inside the cache directory
        rm -rf "$cache_dir"/*
        echo "Cache cleared: $cache_dir"
    else
        echo "Cache directory not found: $cache_dir"
    fi
}

# Function to create directory if it doesn't exist
create_directory() {
    directory="$1"
    
    # Check if directory exists, if not create it
    if [ ! -d "$directory" ]; then
        mkdir -p "$directory"
        echo "Created directory: $directory"
    fi
}

# Function to copy checkpoints
copy_checkpoints() {
    checkpoint_dir="data/generative_re_model_storage_azure/$sacred_run_number/checkpoints"
    mkdir -p "$checkpoint_dir"

    for checkpoint_dir_path in Generative-re-tests/results/checkpoint-*; do
        if [ -d "$checkpoint_dir_path" ]; then
            cp -r "$checkpoint_dir_path" "$checkpoint_dir"
            echo "Copied: $checkpoint_dir_path to $checkpoint_dir"
        fi
    done
}

# Function to copy latest sacred run
copy_latest_sacred_run() {
    latest_sacred_run=$(ls -td Generative-re-tests/sacred_runs/"$sacred_run_number"/*/ | head -n 1)
    if [ -n "$latest_sacred_run" ]; then
        cp -r "$latest_sacred_run" "data/generative_re_model_storage_azure/$sacred_run_number"
        echo "Copied latest sacred run: $latest_sacred_run"
    else
        echo "No sacred run found for sacred_run_number: $sacred_run_number"
    fi
}

# Main function
main() {
    clear_cache

    # Check if directory exists, if not create it
    create_directory "data/generative_re_model_storage_azure/$sacred_run_number"

    # Run loop for specified run_time
    while [ $(($(date +%s) - start_time)) -lt "$run_time" ]; do
        copy_checkpoints
        sleep 300 # Sleep for 5 minutes (300 seconds)
    done

    copy_latest_sacred_run

    end_time=$(date +%s)
    elapsed_time=$((end_time - start_time))
    echo "Script ran for $((elapsed_time / 3600)) hours, $(((elapsed_time / 60) % 60)) minutes, and $((elapsed_time % 60)) seconds."
}

# Execute main function
main
