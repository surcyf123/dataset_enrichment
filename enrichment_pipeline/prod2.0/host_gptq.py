import sys
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import EngineArgs, LLMEngine, SamplingParams
import time
from typing import Optional

# Configuration
model_directory = sys.argv[1].replace("~", "/root/")
port = int(sys.argv[2])
gpu_id = int(sys.argv[3])
gpu_type = sys.argv[4]
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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
    n: int = 1
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 80
    presence_penalty: float = 1.0
    frequency_penalty: float = 1.0
    best_of: Optional[int] = None
    use_beam_search: bool = False
    num_tokens: Optional[int] = None

class ResponseModel(BaseModel):
    response: list
    model: str
    tokens_per_second: float

@app.post('/generate', response_model=ResponseModel)
def generate_text(data: RequestModel):
    time_begin = time.time()
    num_tokens = data.num_tokens or DEFAULT_TOKENS.get(gpu_type)
    if not num_tokens:
        raise HTTPException(status_code=400, detail=f"Invalid gpu_type: {gpu_type}")

    sampling_params = SamplingParams(
        n=data.n,
        max_tokens=num_tokens,
        temperature=data.temperature,
        top_p=data.top_p,
        top_k=data.top_k,
        presence_penalty=data.presence_penalty,
        frequency_penalty=data.frequency_penalty,
        best_of=data.best_of,
        use_beam_search=data.use_beam_search
    )

    request_id = "api_request"
    engine.add_request(request_id, data.prompt, sampling_params)

    finished = False
    while not finished:
        request_outputs = engine.step()
        for request_output in request_outputs:
            if request_output.request_id == request_id and request_output.finished:
                finished = True
                responses = [output.text for output in request_output.outputs]

    time_end = time.time()
    t_per_s = (num_tokens * len(responses)) / (time_end - time_begin)

    return {"response": responses, "model": model_directory, "tokens_per_second": t_per_s}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
