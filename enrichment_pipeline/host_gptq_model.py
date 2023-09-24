import sys
# model_name_or_path = "seonglae/opt-125m-4bit-gptq"
# model_basename = "gptq_model-4bit-128g"


model_name_or_path = sys.argv[1]
local_port = sys.argv[2]
gpuid = str(sys.argv[3])

import threading, time
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

def generate_output(text,max_new_tokens,temperature,top_p,top_k,repetition_penalty,stop_tokens):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(f"cuda:{gpuid}")
    tokens = model.generate(
        inputs=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p = top_p,
        top_k = top_k,
        repetition_penalty = repetition_penalty,
        stopping_criteria=convert_stopwords_to_ids(stop_tokens),
        do_sample=True)
    return tokenizer.decode(tokens[0], skip_special_tokens=True)

app = Flask(__name__)
import re

@app.route('/generate', methods=['POST'])
def generate_text():
    data=request.json
    # Get the hyperparameters and prompt from request
    text = generate_output(
        data['prompt'],
        data['max_new_tokens'],
        data['temperature'],
        data['top_p'],
        data['top_k'],
        data['repetition_penalty'],
        data.get('stopwords', []))
    
    # Remove the prompts from the output
    pruned_prompt = re.sub('<\|.*?\|>', '', data['prompt'])
    text = text.replace(pruned_prompt, "")
    
    # Remove StopToken from the Generation
    for stop in data.get('stopwords', []):
        text = text.replace(stop, "")
    
    return jsonify({'text': text})

def run_app():
    app.run(debug=False, port=local_port)

if __name__ == '__main__':
    t = threading.Thread(target=run_app)
    t.start()
    # Give Flask some time to start
    time.sleep(60)
    print("3ade9fc2-84d5-4a25-8aca-a19f5f301a1d")
