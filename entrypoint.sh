#!/bin/sh

set -e  # Exit immediately if a command exits with a non-zero status

# Ensure the dataset directory exists
mkdir -p /app/Psyche/data/finetune-10bt

# Check if the directory is empty and download the dataset if needed
if [ -z "$(ls -A /app/Psyche/data/finetune-10bt)" ]; then
    echo "Downloading dataset..."
    wget -O /app/Psyche/data/finetune-10bt/00002_00001_shuffled.ds \
         https://huggingface.co/datasets/emozilla/fineweb-10bt-tokenized-datatrove-llama2/resolve/main/00002_00001_shuffled.ds
else
    echo "Dataset already exists. Skipping download."
fi

# Execute the CMD arguments
exec "$@"
