# %%
"""
git clone https://github.com/chu-tianxiang/vllm-gptq.git
cd vllm-gptq/
pip3 install -e .
pip3 install optimum auto-gptq
"""
import sys
from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
model_name_or_path = sys.argv[1]
local_port = sys.argv[2]
gpuid = str(sys.argv[3])

os.environ["CUDA_VISIBLE_DEVICES"] = gpuid

llm = LLM(model=model_name_or_path)


        
        
# TODO: Repetition Penalty is now split into presence penalty and frequency penalty
def generate_output(prompt,max_new_tokens,temperature,top_p,top_k,repetition_penalty,stop_tokens):
    sampling_params = SamplingParams(temperature=temperature,
                                     top_p=top_p,
                                     top_k=top_k,
                                     max_tokens=max_new_tokens,
                                     stop=stop_tokens)
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text


app = Flask(__name__)

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

    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(debug=False, port=local_port)