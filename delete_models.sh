#!/bin/bash

# Define the directory to work in
directory="results"

# Check if the directory exists
if [ -d "$directory" ]; then
    # Navigate to the directory
    cd "$directory" || exit

    # Delete all directories within the current directory
    find . -maxdepth 1 -type d -exec rm -rf {} \;
    echo "All directories in \"$directory\" have been deleted."
else
    echo "Directory \"$directory\" does not exist."
fi
