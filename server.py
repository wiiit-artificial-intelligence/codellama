# server.py
from flask import Flask, request, Response, jsonify
import time
from multiprocessing import Queue
import torch.multiprocessing as multiprocessing

from typing import Optional
import fire
from llama import Llama


PRINT_RESULTS = True

def token_generator(token_str, token_queue, interval=1.0):
    token_stream = token_str.split()

    for t in token_stream:
        print(f"Putting {t}")
        token_queue.put(t)
        time.sleep(interval)
    
    token_queue.put(None)

# The token_queue is used by the model to put generated token into the generation loop
# It could be used when multiprocessing. An alternative is that the model yields tokens.

# token_queue = Queue()
token_queue = None

def build(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 256,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
    device: Optional[str] = 'cuda'
):
    print("Building Llama Generator")
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        device=device,
        token_queue=token_queue
    )

    generator.temperature = temperature
    generator.top_p = top_p

    if max_gen_len is None:
        max_gen_len = max_seq_len - 1
    generator.max_gen_len = max_gen_len

    return generator


app = Flask(__name__)


@app.route('/tokenizate-stream', methods=['POST'])
def process_text():
    try:
        text = request.data.decode('utf-8')  # Get the text from the request body
        words = text.split()  # Split the text into words

        def generate_words():
            for word in words:
                yield word + '\n'  # Send each word as a separate line
                time.sleep(0.5)

        return Response(generate_words(), content_type='text/plain')
    
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/completion-stream-sim', methods=['POST'])
def completion_sim():
    try:
        prompts = request.get_json()
        
        if prompts is None:
            return jsonify({'error': 'Missing or invalid input'}), 400
        
        print(prompts)

        token_queue = Queue()
        
        try:
            multiprocessing.set_start_method('spawn')
        except:
            pass

        generator_process = multiprocessing.Process(target=token_generator, args=(prompts[0], token_queue))
        generator_process.start()

        def extract_tokens(token_queue):
            while True:
                token = token_queue.get()
                if token is None:
                    break

                print(f"extracting {token}")

                yield token
                del token

        return Response(extract_tokens(token_queue), content_type='text/plain')

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/completion-stream', methods=['POST'])
def completion_stream():
    try:
        # Get the input string from the request body
        prompts = request.get_json()
        
        if prompts is None:
            return jsonify({'error': 'Missing or invalid input'}), 400
        
        print(prompts)
 
        def extract_tokens(prompts):
            prompt_tokens = [generator.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
            token_stream = []
            result = ""

            # enabling yield_token converts generate method into a generator of tokens
            # an iteration is needed to get each token
            for token in generator.generate_token(
                prompt_tokens=prompt_tokens,
                max_gen_len=generator.max_gen_len,
                temperature=generator.temperature,
                top_p=generator.top_p,
                logprobs=False,
                echo=False):

                prev_len = len(result)

                token_stream.append(token[0])
                result = generator.tokenizer.decode(token_stream)
                
                # send the new part of decoded tokens string
                diff_len = len(result) - prev_len
                yield result[-diff_len:]

            if PRINT_RESULTS:
                    print(result)

        return Response(extract_tokens(prompts), content_type='text/plain')

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/instructions', methods=['POST'])
def instructions():
    try:
        instructions = request.get_json()
        print(instructions)

        if instructions is None:
            return jsonify({'error': 'Missing or invalid input'}), 400
        
        results = generator.chat_completion(
            instructions,  # type: ignore
            max_gen_len=generator.max_gen_len,
            temperature=generator.temperature,
            top_p=generator.top_p,
        )

        if PRINT_RESULTS:
            for instruction, result in zip(instructions, results):
                for msg in instruction:
                    print(f"{msg['role'].capitalize()}: {msg['content']}\n")
                print(f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}")
                print("\n==================================\n")
        
        return jsonify({'message': 'success', 'results': results}), 200
        
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
    
@app.route('/infilling', methods=['POST'])
def infilling():
    try:
        prompts = request.get_json()
        print(prompts)

        if prompts is None:
            return jsonify({'error': 'Missing or invalid input'}), 400
        
        prefixes = [p.split("<FILL>")[0] for p in prompts]
        suffixes = [p.split("<FILL>")[1] for p in prompts]

        results = generator.text_infilling(
            prefixes=prefixes,
            suffixes=suffixes,
            max_gen_len=generator.max_gen_len,
            temperature=generator.temperature,
            top_p=generator.top_p,
        )

        if PRINT_RESULTS:
            for prompt, result in zip(prompts, results):
                print("\n================= Prompt text =================\n")
                print(prompt)
                print("\n================= Filled text =================\n")
                print(result["full_text"])
            
        return jsonify({'message': 'success', 'results': results}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/completion', methods=['POST'])
def completion():
    try:
        prompts = request.get_json()
        print(prompts)

        if prompts is None:
            return jsonify({'error': 'Missing or invalid input'}), 400

        results = generator.text_completion(
            prompts,
            max_gen_len=generator.max_gen_len,
            temperature=generator.temperature,
            top_p=generator.top_p,
        )

        if PRINT_RESULTS:
            for prompt, result in zip(prompts, results):
                print(prompt)
                print(f"> {result['generation']}")
                print("\n==================================\n")

        return jsonify({'message': 'success', 'results': results}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    generator = fire.Fire(build)
    print("Code Llama model loaded successfully")
    print(f"Device = {generator.device}")
    
    app.run(host='0.0.0.0', port=5000)
    # app.run(debug=True)