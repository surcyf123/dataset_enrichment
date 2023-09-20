import sys
import os
from flask import Flask, request, jsonify
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

model_directory = sys.argv[1]
port = int(sys.argv[2])
gpu_id = int(sys.argv[3])
gpu_type = sys.argv[4]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

# Fixing the model_directory assignment
model_directory = model_directory.replace("~", "/root")

sys.path.append("/root/")
from flask import Flask, request, jsonify

config = ExLlamaV2Config()
config.model_dir = model_directory
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)
model.load([18, 24])
tokenizer = ExLlamaV2Tokenizer(config)
generator = ExLlamaV2BaseGenerator(model, None, tokenizer)  # Initialize with no cache for now
generator.warmup()


def generate_output(text: str, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stopwords, num_completions):

    # Initialize the cache inside the function based on num_completions
    cache = ExLlamaV2Cache(model, batch_size=num_completions)
    generator.cache = cache  # Assign the new cache to the generator

    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = temperature
    settings.top_k = top_k
    settings.top_p = top_p
    settings.token_repetition_penalty = repetition_penalty
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
    settings.token_bias = None
    
    prompts = [text] * num_completions
    
    try:
        full_texts = generator.generate_simple(prompts, settings, max_new_tokens)
        outputs = [full_text[len(text):] for full_text in full_texts]
        time_end = time.time()
        t_per_s = (max_new_tokens * num_completions) / time_total
        return outputs, t_per_s

    except Exception as e:
        print(f"Error during generation: {e}")
        return [], 0


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
        data.get('temperature', 0.9),
        data.get('top_p', 1.0),
        data.get('top_k', 80),
        data.get('repetition_penalty', 1.0),
        data.get('stopwords', []),
        num_responses
    )

    return jsonify({'response': responses, "model": model_directory, "tokens_per_second": t_per_s})


if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=port)