import torch
from typing import Dict, Tuple, List
from flask import Flask, request, jsonify
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM
from torchmetrics.functional import pairwise_cosine_similarity
import torch.nn.functional as F
from abc import abstractmethod


# TODO 
# look into optimizations for the reward models (quantization, clearning cache, make more memory efficient/faster)
# implement nsfw filter
# how to fix the initial high variability of normalization after reset because count = 0?
# improve loading speed of the models (also through quantization? test how quant models affect scores and what the variability is)
# add more logging for checks/debugging; try, excepts; improved error codes for API call fails
# organize dict into reward models and masks
# add diversity model



class BaseRewardModel:
    
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    def __init__(self) -> None:
        self.mean, self.var = 0.0, 0.0

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        rewards = rewards - self.mean
        if self.var > 0:
            rewards /= torch.sqrt(torch.tensor(self.var))
        rewards = 0.5 * (1 + torch.erf(rewards / torch.sqrt(torch.tensor([2.0])).to(rewards.device)))
        return rewards

    def apply(self, prompt: str, responses: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        successful_completions = responses
        successful_rewards = self.get_rewards(prompt, successful_completions)
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)
        
        return successful_rewards, successful_rewards_normalized


class RelevanceRewardModel(BaseRewardModel):
    @property
    def name(self) -> str: 
        return "Relevance Model"
    
    def __init__(self, device: str, models: List[BaseRewardModel]):
        super().__init__()
        self.device = device
        self.models = models
        self.bounds = [-0.0246, 0.3]  # thresholds for the models
    
    def get_rewards(self, prompt: str, completions: List[str]) -> Tuple[torch.FloatTensor, Dict[str, Dict[str, float]]]:
        total_rewards = torch.ones(len(completions), dtype=torch.float32).to(self.device)
        individual_scores = {}
        
        for model in self.models:
            scores = model.get_rewards(prompt, completions)
            threshold = self.bounds[self.models.index(model)]
            binary_scores = [1.0 if score > threshold else 0.0 for score in scores]
            
            individual_scores[model.name] = {"Raw": scores.tolist(), "Binary": binary_scores}
            
            # If any model produces a binary score of 0 for a completion, set its total reward to 0.
            for i, bin_score in enumerate(binary_scores):
                if bin_score == 0.0:
                    total_rewards[i] = 0.0

        return total_rewards, individual_scores

    def apply(self, prompt: str, completions: List[str]) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        total_scores, individual_scores = self.get_rewards(prompt, completions)
        binary_scores = (total_scores > 0.5).float().tolist()  # Convert tensor to list
        
        # Update individual_scores with Binary scores
        for model in self.models:
            threshold = self.bounds[self.models.index(model)]
            individual_scores[model.name]["Binary"] = 1.0 if individual_scores[model.name]["Raw"][0] > threshold else 0.0
        
        return total_scores, binary_scores, individual_scores


class BertRelevanceRewardModel(BaseRewardModel):
    relevance_model_path = "bert-base-uncased"
    
    @property
    def name(self) -> str:
        return "Bert"
    
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(BertRelevanceRewardModel.relevance_model_path)
        self.model = AutoModel.from_pretrained(BertRelevanceRewardModel.relevance_model_path).to(self.device)

    def get_embedding(self, message: str) -> torch.FloatTensor:
        encoded_input = self.tokenizer(message, padding=True, truncation=True, return_overflowing_tokens=True, return_tensors="pt").to(self.device)
        _ = encoded_input.pop("overflow_to_sample_mapping")
        with torch.no_grad():
            embeddings = self.model(**encoded_input)
        return torch.mean(F.normalize(mean_pooling(embeddings, encoded_input["attention_mask"]), p=2, dim=1), dim=0)
    
    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        return torch.tensor([self.reward_single(prompt, completion) for completion in completions], dtype=torch.float32).to(self.device)

    def reward_single(self, prompt: str, completion: str) -> float:
        completion_embedding = self.get_embedding(completion)
        prompt_embedding = self.get_embedding(prompt)
        return float(-((completion_embedding - prompt_embedding)**2).mean()**0.5)

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards

class MpnetRelevanceModel(BaseRewardModel):
    diversity_model_path = "sentence-transformers/all-mpnet-base-v2"
    
    @property
    def name(self) -> str:
        return "MPNet"
    
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MpnetRelevanceModel.diversity_model_path)
        self.model = AutoModel.from_pretrained(MpnetRelevanceModel.diversity_model_path).to(self.device)

    def get_embeddings(self, sentences: List[str]) -> torch.FloatTensor:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**encoded_input)
        return F.normalize(mean_pooling(embeddings, encoded_input["attention_mask"]), p=2, dim=1)
    
    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        return torch.tensor([self.reward_single(prompt, completion) for completion in completions], dtype=torch.float32).to(self.device)

    def reward_single(self, prompt: str, completion: str) -> torch.FloatTensor:
        embeddings = self.get_embeddings([completion])
        prompt_embed = self.get_embeddings([prompt])
        similarity = pairwise_cosine_similarity(prompt_embed, embeddings)
        return torch.abs(similarity)

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards


class OpenAssistantRewardModel(BaseRewardModel):
    reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    @property
    def name(self) -> str:
        return "RLHF"

    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.mean, self.var = 0.75, 1.69
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
        self.mean, self.var = 2.91, 13.35
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
        self.mean, self.var = -11.78, 4.36
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
        self.masking_functions = [
        RelevanceRewardModel(device=self.device, models=[BertRelevanceRewardModel(device=self.device), MpnetRelevanceModel(device=self.device)]),
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

        # 2. Calculate overall mask product
        mask_product = 1.0
        for masking_fn in self.masking_functions:
            _, _, individual_scores = masking_fn.apply(prompt, [completion])
            for model, scores in individual_scores.items():
                completion_results[model] = [scores["Raw"][0], scores["Binary"]]
                mask_product *= scores["Binary"]

        # 3. Final total reward
        total_reward = total_weighted_rewards * mask_product

        return completion_results, total_reward


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=30000, type=int)
    parser.add_argument("--gpu", default=0, type=int)
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
    rw = RewardEndpoint(gpu_id=args.gpu)

    # Testing on launch
    prompt = "Given the historical significance and global influence of various European countries, it's essential to have basic knowledge of their capitals. Keeping that in mind, can you determine the capital of France? The capital of France is"
    completions = [
    "london", "Given the historical significance and global influence of various European countries, the capital of France is most certainly Paris", "Berlin"
    ]
    resulting_dict = rw.calculate_total_reward(prompt, completions)
    print(resulting_dict)

    app.run(host="0.0.0.0", port=args.port, threaded=False, debug=False)
