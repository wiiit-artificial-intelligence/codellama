# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire

from llama import Llama
from multiprocessing import Queue
import torch.multiprocessing as multiprocessing


def print_stream(token_queue, tokenizer=None):
    token_stream = []

    def up_n(n):
        return f"\x1B[{n}A"
    def down_n(n):
        return f"\x1B[{n}B"
    CLR = "\x1B[0K"

    while True:
        token = token_queue.get()
        if token is None:
            break

        token_stream.append(token[0])
        del token

        # print(token_stream[-1])
        if tokenizer is not None:
            msj = tokenizer.decode(token_stream) + f"{CLR}\n"
            nlines = msj.count('\n')
            print(msj, end=up_n(nlines), flush=True)

    print(msj+'\n', flush=True)
    
def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    device: Optional[str] = 'cuda'
):
    token_queue = Queue()

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        token_queue=token_queue
    )

    tokenizer = generator.tokenizer

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        """\
import socket

def ping_exponential_backoff(host: str):"""
    ]

    print(prompts)

    try:
        multiprocessing.set_start_method('spawn')
    except:
        pass

    print("Creating processes")
    print_process = multiprocessing.Process(target=print_stream, args=(token_queue, tokenizer))

    print("Starting processing")
    print_process.start()

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    # End of stream
    token_queue.put(None)
    print_process.join()
    print('\n')

    # print(results)


if __name__ == "__main__":
    fire.Fire(main)

