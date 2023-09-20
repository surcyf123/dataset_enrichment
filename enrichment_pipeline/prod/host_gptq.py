import sys
import os
import torch
import time 
model_directory = sys.argv[1]
port = int(sys.argv[2])
gpu_id = int(sys.argv[3])
gpu_type = sys.argv[4]

# Set CUDA environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
print("Set CUDA_VISIBLE_DEVICES to:", os.environ["CUDA_VISIBLE_DEVICES"])
# device = torch.device(f"cuda:{gpu_id}")

# Ensure model_directory is correctly set
model_directory.replace("~", "/root")

# Add root to system path
sys.path.append("/root/")

from flask import Flask, request, jsonify
from exllamav2 import (ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer)
from exllamav2.generator import (ExLlamaV2BaseGenerator, ExLlamaV2Sampler)

# Configure and load model
config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()
model = ExLlamaV2(config)
print("Loading model:", model_directory)
model.load([18, 24])
tokenizer = ExLlamaV2Tokenizer(config)

# Initialize cache with default size of 3 for num_completions
cache = ExLlamaV2Cache(model, batch_size=3)
generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Warm up the model
generator.warmup()

def generate_output(text, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stopwords, num_completions):
    """Generate text completions based on the given settings."""
    if num_completions != 3:
        cache = ExLlamaV2Cache(model, batch_size=num_completions)
        generator.cache = cache

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.token_repetition_penalty = repetition_penalty
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

    prompts = [text] * num_completions
    time_begin = time.time()
    full_texts = generator.generate_simple(prompts, settings, max_new_tokens)
    outputs = [full_text[len(text):] for full_text in full_texts]

    time_taken = time.time() - time_begin
    t_per_s = (max_new_tokens * num_completions) / time_taken

    return outputs, t_per_s

# ... [rest of the Flask code]



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
    num_responses = data.get('num_responses', 3)

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
