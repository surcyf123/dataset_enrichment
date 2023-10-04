import torch
from typing import Dict, Tuple, List, Union, Optional
from flask import Flask, request, jsonify
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForCausalLM
from torchmetrics.functional import pairwise_cosine_similarity
import torch.nn.functional as F
from abc import abstractmethod
import time
from collections import defaultdict
import traceback
import os
import gc
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# TODO
# check quantized version of cerebras
# figure out tokenization glitch for DPO model (slight difference on batch vs single processing)
# organize dict into reward models and masks - low priority

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=30000, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    return parser.parse_args()


class BaseRewardModel:
    sqrt_of_2: torch.FloatTensor

    @property
    @abstractmethod
    def name(self) -> str: ...
    
    def __init__(self):
        self.sqrt_of_2 = torch.sqrt(torch.tensor([2.0])).to(self.device)
        # Ensure these tensors are on the right device too
        self.mean = torch.tensor([0.0]).to(self.device)
        self.var = torch.tensor([0.0]).to(self.device)
    
    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        # 1. Subtract the mean.
        rewards.sub_(self.mean)
        
        # 2. If the variance is non-zero, divide by the square root of the variance.
        if self.var.item() != 0: 
            rewards.div_(torch.sqrt(self.var))
        
        # 3. Apply the error function transformation.
        rewards = torch.erf(rewards.div(self.sqrt_of_2))
        
        # 4. Scale and translate to the [0, 1] range.
        rewards.mul_(0.5).add_(0.5)
        return rewards
    
    def apply(self, prompt: str, responses: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        successful_rewards = self.get_rewards(prompt, responses)
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)

        return successful_rewards, successful_rewards_normalized

class RelevanceRewardModel(BaseRewardModel):

    @property
    def name(self) -> str:
        return "Relevance Model"
    
    def __init__(self, device: str, models: List[BaseRewardModel]):
        self.device = device
        super().__init__()
        self.models = models
        self.bounds = torch.tensor([-0.0246, 0.3], dtype=torch.float32).to(device)
    
    def get_rewards(self, prompt: str, completions: List[str]) -> Tuple[torch.FloatTensor, Dict[str, Dict[str, float]]]:
        total_rewards = torch.ones(len(completions), dtype=torch.float32).to(self.device)
        individual_scores = {}

        for idx, model in enumerate(self.models):
            scores = model.get_rewards(prompt, completions).to(self.device)
            binary_scores = (scores > self.bounds[idx]).float().squeeze()
            individual_scores[model.name] = {"Raw": scores.tolist(), "Binary": binary_scores.tolist()}
            
            # If any of the completions has a binary score of 0, set the corresponding total reward to 0
            total_rewards *= binary_scores

        return total_rewards, individual_scores

    def apply(self, prompt: str, completions: List[str]) -> Tuple[torch.FloatTensor, Dict[str, float]]:
        total_scores, individual_scores = self.get_rewards(prompt, completions)
        binary_scores = (total_scores > 0.5).float().tolist()

        return total_scores, binary_scores, individual_scores

class BertRelevanceRewardModel(BaseRewardModel):
    relevance_model_path = "bert-base-uncased"

    @property
    def name(self) -> str:
        return "Bert"

    def __init__(self, device: str):
        self.device = device
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(BertRelevanceRewardModel.relevance_model_path)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        assert self.tokenizer.pad_token is not None, "Tokenizer's pad token not set!"
        self.model = AutoModel.from_pretrained(BertRelevanceRewardModel.relevance_model_path).half().to(self.device)

    def get_embedding(self, message: Union[str, List[str]]) -> torch.FloatTensor:
        if isinstance(message, str):
            message = [message]
        encoded_input = self.tokenizer(message, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**encoded_input)[0]
        return F.normalize(mean_pooling(embeddings, encoded_input["attention_mask"]), p=2, dim=1)

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        completion_embeddings = self.get_embedding(completions)
        prompt_embedding = self.get_embedding(prompt).repeat(len(completions), 1)
        return (-((completion_embeddings - prompt_embedding)**2).mean(1)**0.5).to(self.device)

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards

class MpnetRelevanceModel(BaseRewardModel):
    diversity_model_path = "sentence-transformers/all-mpnet-base-v2"

    @property
    def name(self) -> str:
        return "MPNet"

    def __init__(self, device: str):
        self.device = device
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MpnetRelevanceModel.diversity_model_path)
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        assert self.tokenizer.pad_token is not None, f"Tokenizer's pad token not set for {self.name}!"
        self.model = AutoModel.from_pretrained(MpnetRelevanceModel.diversity_model_path).half().to(self.device)

    def get_embeddings(self, sentences: Union[str, List[str]]) -> torch.FloatTensor:
        if isinstance(sentences, str):
            sentences = [sentences]
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            embeddings = self.model(**encoded_input)[0]
        return F.normalize(mean_pooling(embeddings, encoded_input["attention_mask"]), p=2, dim=1)

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        embeddings = self.get_embeddings(completions)
        prompt_embed = self.get_embeddings(prompt).repeat(len(completions), 1)
        similarity_matrix = pairwise_cosine_similarity(prompt_embed, embeddings)
        similarity = torch.diag(similarity_matrix)
        
        return torch.abs(similarity).to(self.device)

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        return rewards

        
class OpenAssistantRewardModel(BaseRewardModel):
    reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"

    @property
    def name(self) -> str:
        return "RLHF"

    def __init__(self, device: str):
        self.device = device
        super().__init__()
        self.mean, self.var = torch.tensor([0.75]).to(self.device), torch.tensor([1.69]).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(OpenAssistantRewardModel.reward_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(OpenAssistantRewardModel.reward_model_name).half().to(self.device)

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        return self.reward_batch(prompt, completions)

    def reward_single(self, prompt: str, completion: str) -> float:
        with torch.no_grad():
            inputs = self.tokenizer(prompt, completion, return_tensors='pt').to(self.device)
            return float(self.model(**inputs).logits[0].cpu().detach())

    def reward_batch(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        with torch.no_grad():
            prompts = [prompt] * len(completions)
            inputs = self.tokenizer(prompts, completions, padding=True, truncation=False, max_length=512, return_tensors='pt').to(self.device)
            return self.model(**inputs).logits.squeeze(1)

class ReciprocateRewardModel(BaseRewardModel):

    reward_model_path = "reciprocate/gpt-j_rm_format-oa"
    revision = "501f895"

    @property
    def name(self) -> str:
        return "Reciprocate"

    def __init__(self, device: str):
        self.device = device
        super().__init__()
        self.mean, self.var = torch.tensor([2.91]).to(self.device), torch.tensor([13.35]).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(ReciprocateRewardModel.reward_model_path, revision=ReciprocateRewardModel.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ReciprocateRewardModel.reward_model_path,
            revision=ReciprocateRewardModel.revision,
            torch_dtype=torch.float16
        ).to(self.device)

    def get_rewards(self, prompt: str, completions: List[str], name=None) -> torch.FloatTensor:
        return self.reward_batch(prompt, completions)

    def reward_batch(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        with torch.no_grad():
            messages = [f"{prompt}</s>{completion}</s>" for completion in completions]
            inputs = self.tokenizer(messages, padding=True, truncation=True, return_tensors="pt").to(self.device)
            logits = self.model(**inputs).logits.squeeze(1)
            return logits

class DirectPreferenceRewardModel(BaseRewardModel):
    reward_model_name = "cerebras/btlm-3b-8k-base"

    @property
    def name(self) -> str:
        return "DPO"

    def __init__(self, device: str):
        self.device = device
        super().__init__()
        self.mean, self.var = torch.tensor([-11.78]).to(self.device), torch.tensor([4.36]).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(DirectPreferenceRewardModel.reward_model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLM.from_pretrained(DirectPreferenceRewardModel.reward_model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def prepare_input(self, prompt: str, completions: List[str]) -> (torch.Tensor, List[Optional[torch.Tensor]]):
        """Tokenizes and prepares the input tensors"""
        prompt_part = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
        input_ids_list = []

        for completion in completions:
            combined = self.tokenizer(prompt + completion, truncation=True, return_tensors="pt").input_ids[0].to(self.device)
            if len(prompt_part) >= self.tokenizer.model_max_length or len(combined) == 0:
                input_ids_list.append(None)
            else:
                input_ids_list.append(combined)

        return prompt_part, input_ids_list

    def reward_batch(self, prompt_part: torch.Tensor, input_ids_list: List[Optional[torch.Tensor]]) -> torch.FloatTensor:
        with torch.no_grad():
            valid_input_ids = [ids for ids in input_ids_list if ids is not None]
            if not valid_input_ids:
                return torch.full((len(input_ids_list),), -11., device=self.device, dtype=torch.float16)

            max_length = max([len(ids) for ids in valid_input_ids])
            padded_input_ids = torch.stack([torch.cat([ids, torch.full((max_length - len(ids),), self.tokenizer.pad_token_id, device=self.device)], 0) for ids in valid_input_ids])
            logits = self.model(padded_input_ids).logits[:, :-1, :]
            logits = logits.log_softmax(-1)
            
            rewards = []
            for idx, combined in enumerate(input_ids_list):
                if combined is None:
                    rewards.append(-11.)
                    continue
                labels = combined[1:]
                per_token_logps = torch.gather(logits[idx], dim=1, index=labels.unsqueeze(1)).squeeze(1)
                reward = per_token_logps.mean()
                if torch.isnan(reward) or torch.isinf(reward):
                    rewards.append(-11.)
                else:
                    rewards.append(reward.item())

            return torch.tensor(rewards, dtype=torch.float16, device=self.device)

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        prompt_part, input_ids_list = self.prepare_input(prompt, completions)
        return self.reward_batch(prompt_part, input_ids_list)

def mean_pooling(model_output, attention_mask, dtype=torch.float32):

    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).to(dtype=dtype)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    mean_embeddings = sum_embeddings / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    return mean_embeddings

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
        # Process all completions at once
        model_scores_batched, total_rewards = self.get_model_scores(prompt, completions)
        
        results = []
        for i, completion in enumerate(completions):
            completion_results = {key: values[i] for key, values in model_scores_batched.items()}
            completion_results["Total Reward"] = total_rewards[i]
            results.append(completion_results)

        return results

    def get_model_scores(self, prompt, completions):
        completion_results = defaultdict(list)
        
        # 1. Calculate total reward from all reward functions
        total_weighted_rewards = torch.zeros(len(completions)).to(self.device)
        for reward_fn in self.reward_functions:
            raw_rewards, normalized_rewards = reward_fn.apply(prompt, completions)
            print(f"{reward_fn.name} Normalized Rewards: {normalized_rewards}")
            weight = self.model_weights[reward_fn.name]  # Direct dictionary access
            total_weighted_rewards.add_(normalized_rewards * weight)
            
            for i, reward in enumerate(raw_rewards):
                completion_results[reward_fn.name].append([raw_rewards[i].item(), normalized_rewards[i].item()])

        # 2. Calculate overall mask product using tensor operations
        binary_scores_list = []
        for masking_fn in self.masking_functions:
            _, _, individual_scores_batched = masking_fn.apply(prompt, completions)
            for model, scores in individual_scores_batched.items():
                binary_scores_list.append(torch.tensor(scores["Binary"]).unsqueeze(1))  # Add a dimension for stacking
                for i in range(len(completions)):
                    completion_results[model].append([scores["Raw"][i], scores["Binary"][i]])
                    
        mask_tensor = torch.cat(binary_scores_list, dim=1)  # Shape: [num_completions, num_models]
        mask_products = mask_tensor.prod(dim=1).to(self.device)  # Compute the product along the model dimension

        # 3. Final total reward for the batch
        total_rewards = (total_weighted_rewards * mask_products).tolist()

        return completion_results, total_rewards

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
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA Out Of Memory error. Clearing GPU memory.")
            torch.cuda.empty_cache()
            gc.collect()
            return jsonify({"error": "Out of GPU memory. Request too large."}), 507  # 507: Insufficient Storage
        else:
            tb = traceback.format_exc()
            print(f"Error: {str(e)}")
            print(tb)
            return str(e), 500

if __name__ == "__main__":
    args = parse_arguments()
    rw = RewardEndpoint(gpu_id=args.gpu)
    prompt = "Given the historical significance and global influence of various European countries, it's essential to have basic knowledge of their capitals. Keeping that in mind, can you determine the capital of France? The capital of France is"
    completions = [
    "london", "Given the historical significance and global influence of various European countries, the capital of France is most certainly Paris", "Berlin"
    ]
    resulting_dict = rw.calculate_total_reward(prompt, completions)
    print(resulting_dict)

    app.run(host="0.0.0.0", port=args.port, threaded=False, debug=False)