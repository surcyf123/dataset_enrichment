import torch
from typing import Dict, Tuple, List
from flask import Flask, request, jsonify
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from torchmetrics.functional import pairwise_cosine_similarity
import torch.nn.functional as F
from abc import abstractmethod
import pprint

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

    def apply(self, prompt: str, responses: List[str], name: str) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
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
        self.bounds = [-0.0246, 0.3]
    
    def get_rewards(self, prompt: str, completions: List[str]) -> Tuple[torch.FloatTensor, Dict[str, Dict[str, float]]]:
        total_rewards = torch.zeros(len(completions), dtype=torch.float32).to(self.device)
        individual_scores = {}
        for model in self.models:
            scores = model.get_rewards(prompt, completions)
            normalized_scores = model.normalize_rewards(scores)
            individual_scores[model.name] = {"Raw": scores.tolist(), "Normalized": normalized_scores.tolist()}
            for i, score in enumerate(scores):
                if score < self.bounds[self.models.index(model)]:
                    total_rewards[i] = 0.0
        
        combined_mask = (total_rewards > 0.5).float()
        return combined_mask, individual_scores

    def apply(self, prompt: str, completions: List[str], name: str) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        masks, _ = self.get_rewards(prompt, completions)
        return masks, (masks > 0.5).float()



class BertRelevanceRewardModel(BaseRewardModel):
    relevance_model_path = "bert-base-uncased"
    
    @property
    def name(self) -> str:
        return "Bert Relevance Model"
    
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


class MpnetRelevanceModel(BaseRewardModel):
    diversity_model_path = "sentence-transformers/all-mpnet-base-v2"
    
    @property
    def name(self) -> str:
        return "MPNet Relevance Model"
    
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

    def reward_single(self, prompt: str, completion: str) -> float:
        embeddings = self.get_embeddings([completion])
        prompt_embed = self.get_embeddings([prompt])
        return torch.abs(pairwise_cosine_similarity(prompt_embed, embeddings)).item()


class OpenAssistantRewardModel(BaseRewardModel):
    reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
    @property
    def name(self) -> str:
        return "RLHF reward model"

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

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class RewardEndpoint:
    def __init__(self, gpu_id):
        self.device = torch.device(f"cuda:{gpu_id}")
        self.model_weights = {"RLHF reward model": 1.0}  # Initial weight setup; can be extended in the future.
        self.reward_functions = [OpenAssistantRewardModel(device=self.device)]
        self.masking_functions = [RelevanceRewardModel(device=self.device, models=[BertRelevanceRewardModel(device=self.device), MpnetRelevanceModel(device=self.device)])]
        
    def calculate_total_reward(self, prompt, completions):
        results = {}
        for completion in completions:
            model_scores, total_reward = self.get_model_scores(prompt, completion)
            
            if isinstance(total_reward, torch.Tensor):  
                total_reward_value = total_reward.item()
            else:
                total_reward_value = total_reward

            results[completion] = {"Total Reward": total_reward_value, **model_scores}
        return results

    def get_model_scores(self, prompt, completion):
        model_scores = {"Rewards": {}}
        total_reward = 1.0 

        for reward_fn in self.reward_functions:
            raw_rewards, normalized_rewards = reward_fn.apply(prompt, [completion], "augment")
            model_name = reward_fn.name
            model_scores["Rewards"][model_name] = [raw_rewards[0].item(), normalized_rewards[0].item()]
            total_reward *= self.model_weights[model_name] * normalized_rewards[0].item()  # Update total reward

        for masking_fn in self.masking_functions:
            raw_scores, binary_scores = masking_fn.apply(prompt, [completion], "augment")
            model_name = masking_fn.name
            model_scores["Rewards"][model_name] = [raw_scores[0].item(), binary_scores[0].item()]
            total_reward *= binary_scores[0].item()  # Multiply by the binary mask

        return model_scores, total_reward




def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8008, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    return parser.parse_args()

app = Flask(__name__)
@app.route("/", methods=["POST"])
def chat():
    data = request.get_json()
    if data.get("verify_token") not in ["SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n"]:
        return jsonify({"error": "Invalid authentication token"}), 401
    prompt, responses = data.get("prompt"), data.get("responses")
    if not (prompt and responses): return "No prompt or responses"
    try:
        rewards, scores = rw.calculate_total_reward(prompt, responses)
        return jsonify({"rewards": rewards.tolist(), "reward_details": scores})
    except Exception as e: return str(e)


# Instantiate the reward models.
bert_model = BertRelevanceRewardModel(device="cuda:0")
mpnet_model = MpnetRelevanceModel(device="cuda:0")
relevance_model = RelevanceRewardModel(device="cuda:0", models=[bert_model, mpnet_model])
open_assistant_model = OpenAssistantRewardModel(device="cuda:0")

# Create RewardEndpoint object
rw = RewardEndpoint(gpu_id=0)

# Define a prompt and a list of completions.
prompt = "Given the historical significance and global influence of various European countries, it's essential to have basic knowledge of their capitals. Keeping that in mind, can you determine the capital of France? The capital of France is"
completions = [
"london", "Paris", "Berlin"
]

# Calculate the rewards and scores for each completion.
resulting_dict = rw.calculate_total_reward(prompt, completions)
print(resulting_dict)


if __name__ == "__main__":
    args = parse_arguments()
    rw = RewardEndpoint(args.gpu)
    app.run(host="0.0.0.0", port=args.port, threaded=False, debug=False)
