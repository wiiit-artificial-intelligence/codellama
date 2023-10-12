# client.py
import requests

def print_stream(token_stream, sep=' ', ret=True):
    def up_n(n):
        return f"\x1B[{n}A"
    def down_n(n):
        return f"\x1B[{n}B"
    CLR = "\x1B[0K"

    msj = sep.join(token_stream) + f"{CLR}\n"
    nlines = msj.count('\n')

    if ret:
        print(msj, end=up_n(nlines), flush=True)
    else:
        print(msj, flush=True)

# url = 'http://localhost:5000/tokenizate'  # Update with your server's URL
# text_to_send = "This is a sample text to be split into words"
# response = requests.post(url, data=text_to_send, stream=True)

# if response.status_code == 200:
#     for word in response.iter_lines():
#         print(word.decode('utf-8'))
# else:
#     print(f"Error: {response.status_code}")

# url = 'http://localhost:5000/completion-sim'  # Update with your server's URL
# prompts = ["This is a sample text to be split into words"]
# response = requests.post(url, json=prompts, stream=True)

# token_stream = []
# if response.status_code == 200:
#     for token in response.iter_content(chunk_size=1024):
#         if token:
#             token_stream.append(token.decode('utf-8'))
#             print_stream(token_stream)
#     print_stream(token_stream, ret=False)
# else:
#     print(f"Error: {response.status_code}")

url = 'http://localhost:5000/completion-stream'  # Update with your server's URL
prompts = [
    "import socket\n\ndef ping_exponential_backoff(host: str):",
    "import argparse\n\ndef main(string: str):\n    print(string)\n    print(string[::-1])\n\nif __name__ == '__main__':"
]
print("Task: completion-stream\n\n")
print(prompts[0])

response = requests.post(url, json=prompts, stream=True)

token_stream = []
if response.status_code == 200:
    for token in response.iter_content(chunk_size=1024):
        if token:
            token_stream.append(token.decode('utf-8'))
            print_stream(token_stream, sep='')
    print_stream(token_stream, sep='', ret=False)
else:
    print(f"Error: {response.status_code}")


