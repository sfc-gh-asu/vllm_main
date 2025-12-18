# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example Python client for OpenAI Chat Completion using vLLM API server
NOTE: start a supported chat completion model server with `vllm serve`, e.g.
    vllm serve meta-llama/Llama-2-7b-chat-hf
"""

import argparse

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

system_content = "You are a helpful assistant. You are given a problem and you need to solve it. You are also given a list of previous problems and their solutions. You need to use the previous problems and solutions to solve the current problem. You need to output the solution to the current problem. " * 512

# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Who won the world series in 2020?"},
#     {
#         "role": "assistant",
#         "content": "The Los Angeles Dodgers won the World Series in 2020.",
#     },
#     {"role": "user", "content": "Where was it played?"},
# ]

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "2+5="},
    {
        "role": "assistant",
        "content": "7",
    },
    # {"role": "user", "content": "4+3="},
    {"role": "user", "content": "8+1="},
    # {"role": "user", "content": "1+3="},
]


def parse_args():
    parser = argparse.ArgumentParser(description="Client for vLLM API server")
    parser.add_argument(
        "--stream", action="store_true", help="Enable streaming response"
    )
    return parser.parse_args()


def main(args):
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    model = models.data[0].id

    # Chat Completion API
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        # max_tokens=256,
        # max_tokens=64,
        # max_tokens=16,
        # max_tokens=4,
        max_tokens=1,
        temperature=0.0,
        stream=args.stream,
    )

    print("-" * 50)
    print("Chat completion results:")
    if args.stream:
        for c in chat_completion:
            print(c)
    else:
        print(chat_completion)
    print("-" * 50)


if __name__ == "__main__":
    args = parse_args()
    main(args)
