import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import json
import csv
import concurrent.futures
import logging
from flask import Flask, request, jsonify
import argparse

# Set the logging level
logging.basicConfig(level=logging.INFO)


class BaseRewardModel:
    """Base class for reward models."""

    def apply(self, prompt: str, responses: List[str]) -> torch.FloatTensor:
        """Apply the model to get rewards."""
        return self.get_rewards(prompt, responses)

    @property
    def name(self) -> str:
        raise NotImplementedError

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        """Get rewards for given prompt and completions."""
        raise NotImplementedError


class OpenAssistantRewardModel(BaseRewardModel):
    reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"

    def __init__(self, device: str):
        super().__init__()
        logging.info("Initializing OpenAssistantRewardModel...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.reward_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.reward_model_name).to(self.device)

    @property
    def name(self) -> str:
        return "RLHF"

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        with torch.no_grad():
            inputs = self.tokenizer([prompt] * len(completions), completions, return_tensors='pt', padding=True, truncation=True).to(self.device)
            logits = self.model(**inputs).logits
            return logits[:, 0]


class ReciprocateRewardModel(BaseRewardModel):
    reward_model_path = "reciprocate/gpt-j_rm_format-oa"
    revision = "501f895"

    def __init__(self, device: str):
        super().__init__()
        logging.info("Initializing ReciprocateRewardModel...")
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.reward_model_path, revision=self.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.reward_model_path, revision=self.revision, torch_dtype=torch.float16).to(self.device)

    @property
    def name(self) -> str:
        return "Reciprocate"

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        with torch.no_grad():
            messages = [f"{prompt}</s>{completion}</s>" for completion in completions]
            inputs = self.tokenizer(messages, return_tensors="pt", padding=True, truncation=True).to(self.device)
            logits = self.model(**inputs).logits
            return logits[:, 0]


class DirectPreferenceRewardModel(BaseRewardModel):
    reward_model_name = "cerebras/btlm-3b-8k-base"
    DEFAULT_REWARD = torch.tensor([-11.0])

    def __init__(self, device: str):
        super().__init__()
        logging.info("Initializing DirectPreferenceRewardModel...")
        self.device = device
        self.penalty = 1.2
        self.tokenizer = AutoTokenizer.from_pretrained(self.reward_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.reward_model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)

    @property
    def name(self) -> str:
        return "DPO"

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        rewards = []
        prompt_part = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
        with torch.no_grad():
            for completion in completions:
                combined = self.tokenizer(prompt + completion, return_tensors="pt").input_ids[0].to(self.device)
                if len(prompt_part) >= self.tokenizer.model_max_length or len(combined) == 0:
                    rewards.append(self.DEFAULT_REWARD)
                    continue

                if self.tokenizer.model_max_length < len(combined):
                    combined = combined[:self.tokenizer.model_max_length]
                labels = combined.clone()
                labels[:len(prompt_part)] = -100
                labels = labels[1:]
                labels[labels == -100] = 0

                logits = self.model(combined.unsqueeze(0)).logits[:, :-1, :]
                logits = logits.log_softmax(-1)
                per_token_logps = torch.gather(logits, dim=2, index=labels.unsqueeze(0).unsqueeze(2)).squeeze(2)
                mask = (labels != -100).unsqueeze(0)
                reward = (per_token_logps[mask]).mean()
                if torch.isnan(reward) or torch.isinf(reward):
                    rewards.append(torch.tensor([-11.0]))
                else:
                    rewards.append(reward)
        return torch.stack(rewards)

class RewardEndpoint:
    """Endpoint to calculate rewards using various reward models."""

    def __init__(self, gpu_ids):
        logging.info("Initializing RewardEndpoint with GPU IDs: %s", gpu_ids)
        self.gpu_ids = gpu_ids
        self.reward_functions = [
            OpenAssistantRewardModel(device=f"cuda:{gpu_ids[0]}"),
            ReciprocateRewardModel(device=f"cuda:{gpu_ids[1]}"),
            DirectPreferenceRewardModel(device=f"cuda:{gpu_ids[2]}"),
        ]

    def calculate_total_reward(self, prompt, completion):
        """Calculate total reward for a given prompt and its completion."""
        model_scores = {}
        for reward_fn in self.reward_functions:
            raw_rewards = reward_fn.apply(str(prompt), [str(completion)])
            score = raw_rewards[0].item()
            model_scores[reward_fn.name] = score
            logging.info(f"Prompt {prompt[:20]}: {reward_fn.name} {score:.4f}")
        return model_scores

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=30000, type=int)
    return parser.parse_args()

app = Flask(__name__)
@app.route("/", methods=["POST"])
def chat():
    data = request.get_json()
    if data.get("verify_token") not in ["SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n"]:
        return jsonify({"error": "Invalid authentication token"}), 401

    prompt, completions = data.get("prompt"), data.get("completions")
    if not (prompt and completions):
        return jsonify({"error": "No prompt or completions provided"}), 400

    try:
        result = rw.calculate_total_reward(prompt, completions)
        return jsonify(result)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    args = parse_arguments()   
    rw = RewardEndpoint(gpu_ids=[0, 1, 2])

    # Testing on launch
    prompt = "Given the historical significance and global influence of various European countries, it's essential to have basic knowledge of their capitals. Keeping that in mind, can you determine the capital of France? The capital of France is"
    completions = [
    "london", "Given the historical significance and global influence of various European countries, the capital of France is most certainly Paris", "Berlin"
    ]
    resulting_dict = rw.calculate_total_reward(prompt, completions)
    print(resulting_dict)

    app.run(host="0.0.0.0", port=args.port, threaded=False, debug=False)
