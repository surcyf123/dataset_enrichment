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
        if new_count and self.count + new_count:
            new_mean, new_var = rewards.mean(), rewards.var()
            new_weight, old_weight = new_count / (self.count + new_count), self.count / (self.count + new_count)
            diff = new_mean - self.mean
            self.mean = new_weight * new_mean + old_weight * self.mean
            self.var = new_weight * new_var + old_weight * self.var + new_weight * old_weight * diff * diff
            self.count = min(self.count_limit, self.count + new_count)
        rewards = (rewards - self.mean) / (torch.sqrt(self.var) if self.var else 1)
        return 0.5 * (1 + torch.erf(rewards / torch.sqrt(torch.tensor(2.0).to(rewards.device))))

    def apply(self, prompt: str, responses: List[str], name: str) -> torch.FloatTensor:
        successful_completions_indices = [idx for idx, resp in enumerate(responses) if resp.is_success]
        successful_completions = [responses[idx].completion.strip() for idx in successful_completions_indices]
        successful_rewards = self.get_rewards(prompt, successful_completions, name)
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)
        filled_rewards = torch.ones(len(responses), dtype=torch.float32) * torch.nan
        filled_rewards_normalized = torch.zeros(len(responses), dtype=torch.float32)
        for idx, reward, reward_normalized in zip(successful_completions_indices, successful_rewards, successful_rewards_normalized):
            filled_rewards[idx], filled_rewards_normalized[idx] = reward, reward_normalized
        return filled_rewards, filled_rewards_normalized

        
class RelevanceRewardModel(BaseRewardModel):
    @property
    def name(self) -> str: 
        return "Relevance Model"
    
    def __init__(self, device: str, models: List[BaseRewardModel]):
        super().__init__()
        self.device = device
        self.models = models
        self.bounds = [-0.0246, 0.3]
    
    def get_rewards(self, prompt: str, completions: List[str]) -> Tuple[torch.FloatTensor, Dict[str, List[float]]]:
        total_rewards = torch.zeros(len(completions), dtype=torch.float32).to(self.device)
        individual_scores = {}
        for model in self.models:
            scores = model.get_rewards(prompt, completions)
            individual_scores[model.name] = scores.tolist()
            for i, score in enumerate(scores):
                if score < self.bounds[self.models.index(model)]:
                    total_rewards[i] = 0.0
        
        return (total_rewards > 0.5).float(), individual_scores

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return (rewards > 0.5).float()


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


class MpnetRelevenceModel(BaseRewardModel):
    diversity_model_path = "sentence-transformers/all-mpnet-base-v2"
    
    @property
    def name(self) -> str:
        return "MPNet Relevance Model"
    
    def __init__(self, device: str):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MpnetRelevenceModel.diversity_model_path)
        self.model = AutoModel.from_pretrained(MpnetRelevenceModel.diversity_model_path).to(self.device)

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
        self.reward_weights = torch.tensor([0.4, 0.6], dtype=torch.float32).to(self.device)
        self.reward_functions = [OpenAssistantRewardModel(device=self.device)]
        self.masking_functions = [RelevanceRewardModel(device=self.device, models=[BertRelevanceRewardModel(device=self.device), MpnetRelevenceModel(device=self.device)])]
        
    def calculate_total_reward(self, prompt, completions):
        results = {}
        
        for completion in completions:
            # Calculate rewards for each completion
            rewards = torch.zeros(1, dtype=torch.float32).to(self.device)
            model_scores = {"Reward Models": {}, "Masking Functions": {}}
            
            # Reward functions
            for weight, reward_fn in zip(self.reward_weights, self.reward_functions):
                reward_i = reward_fn.get_rewards(prompt, [completion])[0]
                rewards += weight * reward_i
                model_scores["Reward Models"][reward_fn.name] = {"Raw": reward_i.item(), "Normalized": reward_i.item()}  
            
            # Masking functions
            for masking_fn in self.masking_functions:
                mask_values, individual_scores = masking_fn.get_rewards(prompt, [completion])
                rewards *= mask_values[0]
                model_scores["Masking Functions"][masking_fn.name] = individual_scores 
                
            results[completion] = {
                "Total Reward": rewards[0].item(),
                **model_scores
            }
        
        return results



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
mpnet_model = MpnetRelevenceModel(device="cuda:0")
relevance_model = RelevanceRewardModel(device="cuda:0", models=[bert_model, mpnet_model])
open_assistant_model = OpenAssistantRewardModel(device="cuda:0")

# Create RewardEndpoint object
rw = RewardEndpoint(gpu_id=0)

# Define a prompt and a list of completions.
prompt = "Given the historical significance and global influence of various European countries, it's essential to have basic knowledge of their capitals. Keeping that in mind, can you determine the capital of France? The capital of France is"
completions = [
"Paris, which is not only the most populous city of France but also a hub for art, fashion, and culture. The city is known for its iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral. Paris is often referred to as 'The City of Light' because it was one of the first cities in the world to have street lighting.",
"Berlin, a city that played a crucial role in world history, especially during the 20th century. Berlin is, in fact, the capital of Germany and is known for its rich history, vibrant culture, and iconic Berlin Wall which once divided the city into East and West.",
"London, an ancient city with a modern twist. From the historic Tower of London to the modern Shard, it offers a blend of old and new. However, London is the capital of the United Kingdom and not France.",
"Madrid, which is the heart of Spain and its largest city. Madrid is known for its grand boulevards, royal palaces, and rich repositories of European art. Yet, it's important to note that Madrid is the capital of Spain, not France.",
]

# Calculate the rewards and scores for each completion.
resulting_dict = rw.calculate_total_reward(prompt, completions)
print(resulting_dict)


if __name__ == "__main__":
    args = parse_arguments()
    rw = RewardEndpoint(args.gpu)
    app.run(host="0.0.0.0", port=args.port, threaded=False, debug=False)
