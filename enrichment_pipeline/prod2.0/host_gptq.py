import os
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import EngineArgs, LLMEngine, SamplingParams 
from typing import Optional
import time

# Configuration Handling using argparse
def get_arguments():
    parser = argparse.ArgumentParser(description="FastAPI server for LLMEngine.")
    parser.add_argument("model_directory", help="Path to the model directory.")
    parser.add_argument("port", type=int, help="Port to run the server on.")
    parser.add_argument("gpu_id", type=int, help="GPU ID to use.")
    parser.add_argument("gpu_type", choices=["3090", "4090"], help="Type of GPU (e.g., 3090, 4090).")
    return parser.parse_args()

args = get_arguments()
model_directory = os.path.expanduser(args.model_directory)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

# Default token counts based on GPU type
DEFAULT_TOKENS = {
    "3090": 130,
    "4090": 170
}

# Initialize the LLMEngine
engine_args = EngineArgs(model=model_directory, quantization="awq")  
engine = LLMEngine.from_engine_args(engine_args) 

app = FastAPI()

class RequestModel(BaseModel):
    prompt: str
    n: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    best_of: Optional[int] = None
    use_beam_search: Optional[bool] = None
    num_tokens: Optional[int] = None

class ResponseModel(BaseModel):
    response: list
    model: str
    tokens_per_second: float

@app.post('/generate', response_model=ResponseModel)
def generate_text(data: RequestModel):
    time_begin = time.time()

    num_tokens = data.num_tokens or DEFAULT_TOKENS.get(args.gpu_type)
    if not num_tokens:
        raise HTTPException(status_code=400, detail=f"Invalid gpu_type: {args.gpu_type}")

    sampling_params = {
        "n": data.n or 1,
        "max_tokens": num_tokens,
        "temperature": data.temperature or 0.9,
        "top_p": data.top_p or 1.0,
        "top_k": data.top_k or 1000,
        "presence_penalty": data.presence_penalty or 1.0,
        "frequency_penalty": data.frequency_penalty or 1.0,
        "use_beam_search": data.use_beam_search or False
    }
    if data.best_of:
        sampling_params["best_of"] = data.best_of

    request_id = "api_request"
    engine.add_request(request_id, data.prompt, SamplingParams(**sampling_params)) 

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

    return {"response": responses, "model": model_directory, "tokens_per_second": t_per_s}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

