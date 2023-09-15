import sys

model = sys.argv[1]
port = int(sys.argv[2])
gpu_id = int(sys.argv[3])
gpu_type = sys.argv[4]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, jsonify
from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)
model.load([18, 24])

tokenizer = ExLlamaV2Tokenizer(config)

cache = ExLlamaV2Cache(model)

# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                            local_files_only=True,
                                           torch_dtype=torch.float16,
                                           trust_remote_code=True,
                                           device_map=f"cuda:{gpuid}")

def generate_output(text: str, max_new_tokens, temperature, top_p, top_k, repetition_penalty,stopwords):

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.token_repetition_penalty = repetition_penalty
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    
    generator.warmup()
    time_begin = time.time()

    output = generator.generate_simple(text, settings, max_new_tokens, seed = 1234)

    time_end = time.time()
    time_total = time_end - time_begin
    
    print(f"Response generated in {time_total:.2f} seconds, {max_new_tokens} tokens, {max_new_tokens / time_total:.2f} tokens/second")
    t_per_s = (max_new_tokens / time_total)
    
    # Decode each item in the batch
    return output,t_per_s

app = Flask(__name__)
import re

@app.route('/generate', methods=['POST'])
def generate_text():
    if gpu_type == "3090":
        num_tokens = 115
    elif gpu_type == "4090":
        num_tokens = 150
    else:
        raise ValueError(f"Invalid gpu_type: {gpu_type}")
    data = request.json
    num_responses = 3  # Number of varied responses for each prompt
    
    # Generate multiple outputs for the prompt
    responses,t_per_s = generate_output(
        data['prompt'],
        num_tokens,
        0.9,
        1.0,
        80,
        1.0,
        [])
    
    
    
    return jsonify({'response': responses, "model": model_directory, "tokens_per_second" : t_per_s})

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=local_port)