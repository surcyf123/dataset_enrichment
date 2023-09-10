import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import json
import csv
import concurrent.futures
import logging
import threading
import time

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
        self.model = AutoModelForSequenceClassification.from_pretrained(self.reward_model_name, torch_dtype=torch.float32).to(self.device)

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

    def reward_single(self, prompt: str, completion: str) -> float:
        with torch.no_grad():
            message = f"{prompt}</s>{completion}</s>"
            inputs = self.tokenizer(message, return_tensors="pt", truncation=True).to(self.device)
            return float(self.model(**inputs)[0].item())

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        return torch.tensor([self.reward_single(prompt, completion) for completion in completions], dtype=torch.float32).to(self.device)


class DirectPreferenceRewardModel(BaseRewardModel):
    reward_model_name = "cerebras/btlm-3b-8k-base"
    DEFAULT_REWARD = torch.tensor([-11.0])

    def __init__(self, device: str):
        super().__init__()
        logging.info("Initializing DirectPreferenceRewardModel...")
        self.device = device
        self.penalty = 1.2
        self.tokenizer = AutoTokenizer.from_pretrained(self.reward_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.reward_model_name, trust_remote_code=True, torch_dtype=torch.float32).to(self.device)

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
                for i in range(len(prompt_part) + 1, len(combined) - 1):
                    logits[:, i, :] = self.logit_penalty(combined[len(prompt_part):i], logits[:, i, :])
                
                logits = logits.log_softmax(-1)
                per_token_logps = torch.gather(logits, dim=2, index=labels.unsqueeze(0).unsqueeze(2)).squeeze(2)
                mask = (labels != -100).unsqueeze(0)
                reward = (per_token_logps[mask]).mean()
                if torch.isnan(reward) or torch.isinf(reward):
                    rewards.append(torch.tensor([-11.0]))
                else:
                    rewards.append(reward)
        return torch.stack(rewards)

    def logit_penalty(self, input_ids: torch.LongTensor, logit: torch.FloatTensor) -> torch.FloatTensor:
        uniques, counts = input_ids.unique(return_counts=True)
        score = torch.gather(logit, 1, uniques.unsqueeze(0))

        score = torch.where(score < 0, score * (self.penalty ** counts), score / (self.penalty ** counts))

        logit.scatter_(1, uniques.unsqueeze(0), score.to(logit.dtype))
        return logit

class RewardEndpoint:
    """Endpoint to calculate rewards using various reward models."""

    def __init__(self, gpu_ids, data):
        self.gpu_ids = gpu_ids
        self.data = data
        self.results = {}
        self.lock = threading.Lock()  # To safely update results from multiple threads

        model_classes = [OpenAssistantRewardModel, ReciprocateRewardModel, DirectPreferenceRewardModel]
        self.threads = [
            threading.Thread(target=self.initialize_and_score, args=(model_class, gpu_id))
            for model_class, gpu_id in zip(model_classes, gpu_ids)
        ]
        for thread in self.threads:
            thread.start()
        for thread in self.threads:
            thread.join()  # Wait for all threads to finish

    def initialize_and_score(self, model_class, gpu_id):
        logging.info(f"Initializing model: {model_class.__name__} on GPU {gpu_id}")
        reward_fn = model_class(device=f"cuda:{gpu_id}")

        # Process the first entry separately to start the timer after it
        first_entry = self.data[0]
        prompt = first_entry["prompt"].strip('\n')
        completion = first_entry["completions"].strip('\n')
        reward_fn.apply(str(prompt), [str(completion)])

        start_time = time.time()
        for idx, entry in enumerate(self.data[1:], 2):  # Starting from the second entry
            prompt = entry["prompt"].strip('\n')
            completion = entry["completions"].strip('\n')
            try:
                raw_rewards = reward_fn.apply(str(prompt), [str(completion)])
                score = raw_rewards[0].item()
                with self.lock:
                    if idx not in self.results:
                        self.results[idx] = {}
                    self.results[idx][reward_fn.name] = score
                logging.info(f"Prompt {idx}: {reward_fn.name} {score:.4f}")
            except torch.cuda.OutOfMemoryError:
                logging.error(f"Ran out of GPU memory on prompt {idx} for model {reward_fn.name}. Skipping this prompt.")
                torch.cuda.empty_cache()
        end_time = time.time()

        elapsed_time = end_time - start_time
        avg_time_per_prompt = elapsed_time / (len(self.data) - 1)  # Subtracting one for the first entry
        logging.info(f"Average time per prompt-completion pair for {model_class.__name__}: {avg_time_per_prompt:.4f} seconds")

file_path = "/root/dataset_enrichment/dataset/benchmarking_completions.json"
with open(file_path, 'r') as f:
    data = json.load(f)

# Main execution flow
endpoint = RewardEndpoint(gpu_ids=[0, 1, 2], data=data)

# Convert the results to a list for CSV writing
csv_rows = []
for idx, scores in endpoint.results.items():
    row_data = {'Prompt Number': idx}
    row_data.update(scores)
    csv_rows.append(row_data)

# Write to CSV
csv_path = "/root/dataset_enrichment/enrichment_pipeline/results/benchmarks/benchmarks0909.csv"
with open(csv_path, 'w', newline='') as csvfile:
    fieldnames = ['Prompt Number', 'RLHF', "Reciprocate", "DPO"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row_data in csv_rows:
        writer.writerow(row_data)