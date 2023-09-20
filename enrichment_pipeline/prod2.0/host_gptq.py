import sys
import os
import time

from flask import Flask, request, jsonify
from vllm import LLM, SamplingParams

model_directory = sys.argv[1]
port = int(sys.argv[2])
gpu_id = int(sys.argv[3])
gpu_type = sys.argv[4]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

print(model_directory)
model_directory.replace("~", "/root")
sys.path.append("/root/")
print(f"sys.path: ", sys.path)

llm = LLM(model=model_directory)

def generate_output(text: str):
    sampling_params = SamplingParams(
        temperature=0.9, top_p=1.0, top_k=80
    )
    num_responses = 1 # Number of varied responses for each prompt
    outputs = llm.generate([text], sampling_params, n=num_responses)
    return [output.outputs[0].text for output in outputs[0]]

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

    time_begin = time.time()
    responses = generate_output(data['prompt'])
    time_end = time.time()
    time_total = time_end - time_begin
    t_per_s = (num_tokens * len(responses)) / time_total

    return jsonify({'response': responses, "model": model_directory, "tokens_per_second": t_per_s})

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=port)
