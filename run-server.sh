CODELLAMA_MODEL=CodeLlama-7b
# CODELLAMA_MODEL=CodeLlama-7b-Python
# CODELLAMA_MODEL=CodeLlama-7b-Instruct

torchrun --nproc_per_node 1 server.py \
    --ckpt_dir $CODELLAMA_MODEL/ \
    --tokenizer_path $CODELLAMA_MODEL/tokenizer.model \
    --max_seq_len 256 \
    --max_batch_size 2 \
    --device cpu