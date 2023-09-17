import sys
import os
import time
import re

model_directory = sys.argv[1]
port = int(sys.argv[2])
gpu_id = int(sys.argv[3])
gpu_type = sys.argv[4]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print(model_directory)
model_directory.replace("~", "/root")
sys.path.append("/root/")
print(f"sys.path: ", sys.path)
from flask import Flask, request, jsonify
from exllamav2 import(ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer,)
from exllamav2.generator import (ExLlamaV2BaseGenerator, ExLlamaV2Sampler)

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)
model.load([18, 24])

tokenizer = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache(model)

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)


def generate_output(text: str, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stopwords, num_completions):

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.token_repetition_penalty = repetition_penalty
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    outputs = []
    time_begin = time.time()

    for _ in range(num_completions):
        generator.warmup()
        output = generator.generate_simple(text, settings, max_new_tokens, seed=None)  # Removed seed for variability
        outputs.append(output)

    time_end = time.time()
    time_total = time_end - time_begin
    t_per_s = (max_new_tokens * num_completions) / time_total

    return outputs, t_per_s


app = Flask(__name__)


@app.route('/generate', methods=['POST'])
def generate_text():
    if gpu_type == "3090":
        num_tokens = 130
    elif gpu_type == "4090":
        num_tokens = 170
    else:
        raise ValueError(f"Invalid gpu_type: {gpu_type}")

    data = request.json
    num_responses = 3  # Number of varied responses for each prompt

    # Generate multiple outputs for the prompt
    responses, t_per_s = generate_output(
        data['prompt'],
        num_tokens,
        0.9,
        1.0,
        80,
        1.0,
        [],
        num_responses
    )

    return jsonify({'response': responses, "model": model_directory, "tokens_per_second": t_per_s})


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=port)
