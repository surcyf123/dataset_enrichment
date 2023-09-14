import sys

model = sys.argv[1]
port = int(sys.argv[2])
gpu_id = int(sys.argv[3])
gpu_type = sys.argv[4]

from flask import Flask, request, jsonify

import accelerate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from typing import List

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                          use_fast=False,
                                          local_files_only=True,
                                          trust_remote_code=True)
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops = []):
        super().__init__()
        self.stops = [stop.to("cuda:{gpuid}") for stop in stops]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


def convert_stopwords_to_ids(stopwords : List[str]):
    stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stopwords]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    return stopping_criteria


model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                            local_files_only=True,
                                           torch_dtype=torch.float16,
                                           trust_remote_code=True,
                                           device_map=f"cuda:{gpuid}")

def generate_output(text: str, num_responses: int, max_new_tokens, temperature, top_p, top_k, repetition_penalty, stop_tokens):
    # Convert the text to input_ids
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(f"cuda:{gpuid}")
    
    # Use num_return_sequences to generate multiple completions in parallel
    tokens = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        stopping_criteria=convert_stopwords_to_ids(stop_tokens),
        do_sample=True,
        num_return_sequences=num_responses
    )
    
    # Decode each item in the batch
    return [tokenizer.decode(token, skip_special_tokens=True) for token in tokens]



app = Flask(__name__)
import re

@app.route('/generate', methods=['POST'])
def generate_text():
    if gpu_type == "3090":
        num_tokens = 200
    elif gpu_type == "4090":
        num_tokens = 250
    else:
        raise ValueError(f"Invalid gpu_type: {gpu_type}")
    data = request.json
    num_responses = 4  # Number of varied responses for each prompt
    
    # Generate multiple outputs for the prompt
    responses = generate_output(
        data['prompt'],
        num_responses,
        num_tokens,
        0.95,
        1.0,
        90,
        1.0,
        []
    )
    
    # Clean up the responses
    for i, response in enumerate(responses):
        pruned_prompt = re.sub('<\|.*?\|>', '', data['prompt'])
        responses[i] = response.replace(pruned_prompt, "")
        
        # Remove StopToken from the Generation
        for stop in data.get('stopwords', []):
            responses[i] = responses[i].replace(stop, "")
    
    return jsonify({'response': responses, "model": model_name_or_path})

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=local_port)