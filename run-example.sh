#!/bin/bash

# Check if there is exactly one argument
if [ $# -lt 1 ]; then
  echo "Usage: $0 <example name> [device (optional)] [model (optional)]" 
  echo "    - example name options: {completion, completion-stream, infilling, instructions}"
  echo "    - device options: {cuda (default), cpu}"
  echo "    - model options: {CodeLlama-7b, CodeLlama-7b-Python, CodeLlama-7b-Instruct}"
  exit 1
fi

# Store the example name in a variable
example="$1"

# device
device="${2:-cuda}"

if [ "$example" == "completion" ]; then
    CODELLAMA_MODEL=${3:-CodeLlama-7b}
    # CODELLAMA_MODEL=${3:-CodeLlama-7b-Python}
    # CODELLAMA_MODEL=${3:-CodeLlama-7b-Instruct}

    echo "Executing $example example with model $CODELLAMA_MODEL over $device"

    torchrun --nproc_per_node 1 example_completion.py \
        --ckpt_dir $CODELLAMA_MODEL/ \
        --tokenizer_path $CODELLAMA_MODEL/tokenizer.model \
        --max_seq_len 128 \
        --max_batch_size 2 \
        --device $device

elif [ "$example" == "completion-stream" ]; then
    CODELLAMA_MODEL=${3:-CodeLlama-7b}
    # CODELLAMA_MODEL=${3:-CodeLlama-7b-Python}
    # CODELLAMA_MODEL=${3:-CodeLlama-7b-Instruct}

    echo "Executing $example example with model $CODELLAMA_MODEL over $device"

    torchrun --nproc_per_node 1 example_completion_stream.py \
        --ckpt_dir $CODELLAMA_MODEL/ \
        --tokenizer_path $CODELLAMA_MODEL/tokenizer.model \
        --max_seq_len 256 \
        --max_batch_size 1 \
        --device $device

elif [ "$example" == "infilling" ]; then
    CODELLAMA_MODEL=${3:-CodeLlama-7b}
    # CODELLAMA_MODEL=${3:-CodeLlama-7b-Python}
    # CODELLAMA_MODEL=${3:-CodeLlama-7b-Instruct}

    echo "Executing $example example with model $CODELLAMA_MODEL over $device"

    torchrun --nproc_per_node 1 example_infilling.py \
        --ckpt_dir $CODELLAMA_MODEL/ \
        --tokenizer_path $CODELLAMA_MODEL/tokenizer.model \
        --max_seq_len 192 \
        --max_batch_size 4 \
        --device $device

elif [ "$example" == "instructions" ]; then
    # CODELLAMA_MODEL=${3:-CodeLlama-7b}
    # CODELLAMA_MODEL=${3:-CodeLlama-7b-Python}
    CODELLAMA_MODEL=${3:-CodeLlama-7b-Instruct}

    echo "Executing $example example with model $CODELLAMA_MODEL over $device"

    torchrun --nproc_per_node 1 example_instructions.py \
        --ckpt_dir $CODELLAMA_MODEL/ \
        --tokenizer_path $CODELLAMA_MODEL/tokenizer.model \
        --max_seq_len 512 \
        --max_batch_size 4 \
        --device $device

else 
    echo "Invalid example name: $example"
    exit 1
fi