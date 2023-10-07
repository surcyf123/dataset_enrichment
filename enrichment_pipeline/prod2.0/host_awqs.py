import subprocess
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from vllm import EngineArgs, LLMEngine, SamplingParams 
import argparse
import uvicorn
import sys
import time
import os
from typing import Optional

args = None
VALID_GPU_TYPES = ["3090", "4090"]
DEFAULT_TOKENS = {
    "3090": 160,
    "4090": 160,
}
MODEL_CHOICES = {
"models8x1":["Huginn-13B-v4-AWQ", "UndiMix-v2-13B-AWQ", "Huginn-13B-v4.5-AWQ", "Huginn-v3-13B-AWQ", "Stheno-Inverted-L2-13B-AWQ", "Speechless-Llama2-Hermes-Orca-Platypus-WizardLM-13B-AWQ", "Speechless-Llama2-13B-AWQ", "Mythical-Destroyer-V2-L2-13B-AWQ"],
"models8x2":["Mythical-Destroyer-L2-13B-AWQ", "MythoBoros-13B-AWQ", "StableBeluga-13B-AWQ", "CodeUp-Llama-2-13B-Chat-HF-AWQ", "Baize-v2-13B-SuperHOT-8K-AWQ", "orca_mini_v3_13B-AWQ", "Chronoboros-Grad-L2-13B-AWQ", "Project-Baize-v2-13B-AWQ"],
"models8x3":["PuddleJumper-13B-AWQ", "Luban-13B-AWQ", "LosslessMegaCoder-Llama2-13B-Mini-AWQ", "Luban-13B-AWQ", "OpenOrca-Platypus2-13B-AWQ", "Llama2-13B-MegaCode2-OASST-AWQ", "Chronos-Hermes-13B-SuperHOT-8K-AWQ", "OpenOrcaxOpenChat-Preview2-13B-AWQ"],
}

def start_model_server(model, port, gpu_id, gpu_type):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu_id))
    command = [
        'pm2', 'start', 'python3',
        '-n', model.split('/')[-1],  # Using the model name as the process name
        '--', sys.argv[0],
        '--model', model,
        '--port', str(port),
        '--gpu_type', gpu_type
    ]
    subprocess.run(command, env=env, check=True)

class RequestModel(BaseModel):
    prompt: str
    n: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    use_beam_search: Optional[bool] = False
    num_tokens: Optional[int] = None

async def process_request(data: RequestModel, gpu_type, model, engine):
    num_tokens = data.num_tokens or DEFAULT_TOKENS.get(gpu_type)
    if not num_tokens:
        raise HTTPException(status_code=400, detail=f"Invalid gpu_type: {gpu_type}")

    sampling_params = {
        "n": data.n or 16,
        "max_tokens": num_tokens,
        "temperature": data.temperature or 0.9,
        "top_p": data.top_p or 1.0,
        "top_k": data.top_k or 1000,
        "presence_penalty": data.presence_penalty or 1.0,
        "frequency_penalty": data.frequency_penalty or 1.0,
    }

    request_id = "api_request"
    engine.add_request(request_id, data.prompt, SamplingParams(**sampling_params)) 

    time_begin = time.time() 
    # Fetching the response from the engine
    responses = []
    finished = False
    while not finished:
        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.request_id == request_id and request_output.finished:
                finished = True
                responses = [output.text for output in request_output.outputs]

    # Calculate tokens_per_second
    time_end = time.time()
    t_per_s = (num_tokens * len(responses)) / (time_end - time_begin)
    return {
        "response": responses,
        "model": "13b model", #replace with 'model'
        "tokens_per_second": t_per_s
    }

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configuration for FastAPI model serving.")
    parser.add_argument("--gpu_type", type=str, required=True, choices=VALID_GPU_TYPES, help="Type of GPU.")
    parser.add_argument("--model_choice", type=str, choices=MODEL_CHOICES.keys(), help="Choice of model group.")
    parser.add_argument("--model", type=str, help="Model directory.")
    parser.add_argument("--port", type=int, help="Port for the FastAPI server.")
    parser.add_argument("--gpu_id", type=int, help="GPU ID to use.")
    return parser.parse_args()

def initialize_engine(model):
    engine_args = EngineArgs(model=model, quantization="awq")
    return LLMEngine.from_engine_args(engine_args)

args = parse_arguments()
engine_instance = initialize_engine(args.model) if args.model else None

app = FastAPI()
@app.post('/generate')
async def generate_text(data: RequestModel):
    return await process_request(data, args.gpu_type, args.model, engine_instance)

def main():
    if args.model:
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    elif args.model_choice:
        for i, model_name in enumerate(MODEL_CHOICES[args.model_choice]):
            start_model_server(f"TheBloke/{model_name}", 30000 + i, i, args.gpu_type)
    else:
        print("Please provide either --model or --model_choice argument.")
        sys.exit(1)

if __name__ == "__main__":
    main()
