import torch
from typing import Dict, Tuple, List
from flask import Flask, request, jsonify
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM
from torchmetrics.functional import pairwise_cosine_similarity
import torch.nn.functional as F
from abc import abstractmethod
import json
import csv

class BaseRewardModel:
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    def __init__(self) -> None:
        self.count, self.mean, self.var = 0, 0.0, 0.0
        self.count_limit = 3000

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        new_count = rewards.numel()
        if 0 < new_count and 0 < self.count + new_count:
            new_mean = rewards.mean()
            new_var = rewards.var(dim=0)
            new_weight = new_count / (self.count + new_count)
            old_weight = self.count / (self.count + new_count)
            diff = new_mean - self.mean
            self.mean = new_weight * new_mean + old_weight * self.mean
            self.var = (new_weight * new_var) + (old_weight * self.var) + (new_weight * old_weight) * diff * diff
            self.count = min(self.count_limit, self.count + new_count)
        rewards = rewards - self.mean
        if self.var > 0:
            rewards /= torch.sqrt(self.var)
        rewards = 0.5 * (1 + torch.erf(rewards / torch.sqrt(torch.tensor([2.0])).to(rewards.device)))
        return rewards

    def compute_statistics_from_file(self, file_path: str) -> None:
        # Load the data from the file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        all_prompts = [entry["prompt"] for entry in data]
        all_completions = [entry["completions"] for entry in data]
        
        # Iterate through the prompts and completions and compute rewards
        all_rewards = []
        for i, (prompt, completion) in enumerate(zip(all_prompts, all_completions)):
            reward = self.get_rewards(prompt, [completion])
            print(f"Processing prompt {i+1} with reward: {reward.item()}")
            all_rewards.append(reward)
        
        # Convert the list of rewards to a tensor
        all_rewards_tensor = torch.cat(all_rewards)  # Concatenate all tensors into one
        
        # Compute and set the base mean and variance
        self.base_mean = all_rewards_tensor.mean().item()
        self.base_var = all_rewards_tensor.var().item()
        
        # Print the final base mean and variance
        print(f"Base Mean: {self.base_mean}")
        print(f"Base Variance: {self.base_var}")

    def apply(self, prompt: str, responses: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        successful_completions = responses
        successful_rewards = self.get_rewards(prompt, successful_completions)
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)
        
        return successful_rewards, successful_rewards_normalized

class OpenAssistantRewardModel(BaseRewardModel):
    reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    @property
    def name(self) -> str:
        return "RLHF"

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(OpenAssistantRewardModel.reward_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(OpenAssistantRewardModel.reward_model_name).to(self.device)
    def reward_single(self, prompt: str, completion: str) -> float:
        with torch.no_grad():
            inputs = self.tokenizer(prompt, completion, return_tensors='pt').to(self.device)
            return float(self.model(**inputs).logits[0].cpu().detach())
    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        return torch.tensor([self.reward_single(prompt, completion) for completion in completions], dtype=torch.float32).to(self.device)

class ReciprocateRewardModel(BaseRewardModel):

    reward_model_path = "reciprocate/gpt-j_rm_format-oa"
    revision = "501f895"

    @property
    def name(self) -> str:
        return "Reciprocate"

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(ReciprocateRewardModel.reward_model_path, revision=ReciprocateRewardModel.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ReciprocateRewardModel.reward_model_path,
            revision=ReciprocateRewardModel.revision,
            torch_dtype=torch.float16
        ).to(self.device)

    def reward_single(self, prompt: str, completion: str) -> float:
        with torch.no_grad():
            message = f"{prompt}</s>{completion}</s>"
            inputs = self.tokenizer(message, return_tensors="pt", truncation=True).to(self.device)
            return float(self.model(**inputs).logits[0].item())

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        return torch.tensor([self.reward_single(prompt, completion) for completion in completions], dtype=torch.float32).to(self.device)

class DirectPreferenceRewardModel(BaseRewardModel):
    reward_model_name = "cerebras/btlm-3b-8k-base"

    @property
    def name(self) -> str:
        return "DPO"

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.penalty = 1.2
        self.tokenizer = AutoTokenizer.from_pretrained(DirectPreferenceRewardModel.reward_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(DirectPreferenceRewardModel.reward_model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)

    def reward_single(self, prompt: str, completion: str) -> float:
        with torch.no_grad():
            combined = self.tokenizer(prompt + completion, return_tensors="pt").input_ids[0].to(self.device)
            prompt_part = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
            
            if len(prompt_part) >= self.tokenizer.model_max_length or len(combined) == 0:
                return -11.  

            if self.tokenizer.model_max_length < len(combined):
                combined = combined[:self.tokenizer.model_max_length]

            labels = combined.clone()
            labels[:len(prompt_part)] = -100
            labels = labels[1:]
            labels[labels == -100] = 0

            logits = self.model(combined.unsqueeze(0)).logits[:, :-1, :]
            logits = logits.log_softmax(-1)
            per_token_logps = torch.gather(logits, dim=2, index=labels.unsqueeze(0).unsqueeze(2)).squeeze(2)
            mask = (labels != -100).unsqueeze(0)  # This will change the mask shape to [1, 42]
            reward = (per_token_logps[mask]).mean()

            if torch.isnan(reward) or torch.isinf(reward):
                return -11.  
            return reward.item()

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        return torch.tensor([self.reward_single(prompt, completion) for completion in completions], dtype=torch.float32).to(self.device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class RewardEndpoint:
    def __init__(self, gpu_id):
        self.device = torch.device(f"cuda:{gpu_id}")
        self.model_weights = {"RLHF": .4, "DPO": .3, "Reciprocate": .3} 
        self.reward_functions = [
            OpenAssistantRewardModel(device=self.device),
            ReciprocateRewardModel(device=self.device),
            DirectPreferenceRewardModel(device=self.device),
        ]

    def calculate_total_reward(self, prompt, completions):
        results = []

        for completion in completions:
            model_scores, total_reward = self.get_model_scores(prompt, completion)
            model_scores["Total Reward"] = total_reward
            results.append(model_scores)

        return results

    def get_model_scores(self, prompt, completion):
        completion_results = {}

        # 1. Calculate total reward from all reward functions
        total_weighted_rewards = 0
        for reward_fn in self.reward_functions:
            raw_rewards, normalized_rewards = reward_fn.apply(prompt, [completion])
            weight = self.model_weights.get(reward_fn.name, 1.0)
            total_weighted_rewards += weight * normalized_rewards[0].item()
            completion_results[reward_fn.name] = [raw_rewards[0].item(), normalized_rewards[0].item()]

        return completion_results, total_weighted_rewards


# Compute statistics from the file
file_path = "/root/dataset_enrichment/dataset/example_answers.json"

# Initialize the RewardEndpoint
endpoint = RewardEndpoint(gpu_id=0)

# Compute statistics for each model
for reward_function in endpoint.reward_functions:
    reward_function.compute_statistics_from_file(file_path)

# Load the data from the file
with open(file_path, 'r') as f:
    data = json.load(f)

# Set up the path to the CSV file and open it for writing
csv_path = "/root/dataset_enrichment/enrichment_pipeline/results/benchmarks/benchmarks0909.csv"

with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['RLHF', "Reciprocate", "DPO", 'Total Reward', 'Prompt', 'Completion']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for entry in data:
        prompt = entry["prompt"].strip('\n')  # Strip newline characters from the prompt
        completions = [comp.strip('\n') for comp in entry["completions"]]  # Strip newline characters from each completion
        for completion in completions:
            rewards = endpoint.calculate_total_reward(prompt, [completion])
            for reward in rewards:
                row_data = {'Prompt': prompt, 'Completion': completion}
                row_data.update(reward)
                writer.writerow(row_data)

print(f"Data saved to {csv_path}")

