import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import json
import csv
import concurrent.futures
import logging

# Set the logging level
logging.basicConfig(level=logging.INFO)


class BaseRewardModel:
    """Base class for reward models."""

    def compute_statistics_from_file(self, file_path: str) -> None:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        all_rewards = [self.get_rewards(entry["prompt"], [entry["completions"]])[0] for entry in data]
        all_rewards_tensor = torch.stack(all_rewards)
        self.base_mean = all_rewards_tensor.mean().item()
        self.base_var = all_rewards_tensor.var().item()

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

    def calculate_total_reward(self, prompt, completion, prompt_number):
        """Calculate total reward for a given prompt and its completion."""
        model_scores = {}
        for reward_fn in self.reward_functions:
            raw_rewards = reward_fn.apply(str(prompt), [str(completion)])
            score = raw_rewards[0].item()
            model_scores[reward_fn.name] = score
            logging.info(f"Prompt {prompt_number}: {reward_fn.name} {score:.4f}")
        return model_scores

# Main execution flow
endpoint = RewardEndpoint(gpu_ids=[0, 1, 2])

file_path = "/root/dataset_enrichment/dataset/benchmarking_completions.json"
with open(file_path, 'r') as f:
    data = json.load(f)

def process_data_on_gpu(data, reward_function):
    results = []
    for idx, entry in enumerate(data, 0):
        prompt = entry["prompt"].strip('\n')
        completion = entry["completions"].strip('\n')
        reward = reward_function.calculate_total_reward(prompt, completion, idx)
        row_data = {'Prompt Number': idx}
        row_data.update(reward)
        results.append(row_data)
    return results

results = process_data_on_gpu(data, endpoint)

csv_path = "/root/dataset_enrichment/enrichment_pipeline/results/benchmarks/benchmarks0909.csv"
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['Prompt Number', 'RLHF', "Reciprocate", "DPO"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row_data in results:
        writer.writerow(row_data)