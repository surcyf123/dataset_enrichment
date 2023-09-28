from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Tuple, Optional
from abc import abstractmethod
import threading
import os
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class BaseRewardModel:
    sqrt_of_2: torch.FloatTensor

    @property
    @abstractmethod
    def name(self) -> str: ...

    def __init__(self):
        self.sqrt_of_2 = torch.sqrt(torch.tensor([2.0])).to(self.device)
        self.mean = torch.tensor([0.0]).to(self.device)
        self.var = torch.tensor([0.0]).to(self.device)

    def normalize_rewards(self, rewards: torch.FloatTensor) -> torch.FloatTensor:
        zero_tensor = torch.tensor([0.0]).to(self.mean.device)
        one_half_tensor = torch.tensor([0.5]).to(self.mean.device)

        rewards.sub_(self.mean)
        if self.var.item() != 0:
            rewards.div_(torch.sqrt(self.var))
        rewards.mul_(one_half_tensor).add_(torch.erf(rewards.div(self.sqrt_of_2)))
        return rewards

    def apply(self, prompt: str, responses: List[str]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        successful_rewards = self.get_rewards(prompt, responses, name)
        successful_rewards_normalized = self.normalize_rewards(successful_rewards)
        return successful_rewards, successful_rewards_normalized

class ReciprocateRewardModelV1(BaseRewardModel):
    reward_model_path: str = "reciprocate/gpt-j_rm_format-oa"
    revision: str = "501f895"

    @property
    def name(self) -> str: 
        return "ReciprocateV1"

    def __init__(self, device: str):
        self.device = device
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            ReciprocateRewardModelV1.reward_model_path,
            revision=ReciprocateRewardModelV1.revision
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ReciprocateRewardModelV1.reward_model_path,
            revision=ReciprocateRewardModelV1.revision,
            torch_dtype=torch.float16
        ).to(self.device)

    def reward_single(self, prompt: str, completion: str, name=None) -> float:
        with torch.no_grad():
            message = f"{prompt}</s>{completion}</s>"
            inputs = self.tokenizer(message, return_tensors="pt", truncation=True).to(self.device)
            return float(self.model(**inputs).logits[0].item())

    def get_rewards(self, prompt: str, completions: List[str], name=None) -> torch.FloatTensor:
        return torch.tensor([self.reward_single(prompt, completion) for completion in completions], dtype=torch.float32).to(self.device)


class ReciprocateRewardModelV2(BaseRewardModel):
    reward_model_path = "reciprocate/gpt-j_rm_format-oa"
    revision = "501f895"

    @property
    def name(self) -> str:
        return "ReciprocateV2"

    def __init__(self, device: str):
        self.device = device
        super().__init__()
        self.mean, self.var = torch.tensor([2.91]).to(self.device), torch.tensor([13.35]).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(ReciprocateRewardModelV2.reward_model_path, revision=ReciprocateRewardModelV2.revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            ReciprocateRewardModelV2.reward_model_path,
            revision=ReciprocateRewardModelV2.revision,
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

class OpenAssistantRewardModelV1(BaseRewardModel):
    reward_model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2"

    @property
    def name(self) -> str: 
        return "RLHFV1"  # Replaced the RewardModelType enum with a hardcoded string for simplicity

    def __init__(self, device: str):
        self.device = device  # Initialize device first
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(OpenAssistantRewardModelV1.reward_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(OpenAssistantRewardModelV1.reward_model_name).to(self.device)

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        return torch.tensor([self.reward_single(prompt, completion) for completion in completions], dtype=torch.float32).to(self.device)

    def reward_single(self, prompt: str, completion: str) -> float:
        with torch.no_grad():
            inputs = self.tokenizer(prompt, completion, return_tensors="pt").to(self.device)
            return float(self.model(**inputs).logits[0].cpu().detach())

class OpenAssistantRewardModelV2(BaseRewardModel):
    reward_model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"

    @property
    def name(self) -> str:
        return "RLHFV2"

    def __init__(self, device: str):
        self.device = device
        super().__init__()
        self.mean, self.var = torch.tensor([0.75]).to(self.device), torch.tensor([1.69]).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(OpenAssistantRewardModelV2.reward_model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(OpenAssistantRewardModelV2.reward_model_name).half().to(self.device)

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

class DirectPreferenceRewardModelV1(BaseRewardModel):
    reward_model_name: str = "cerebras/btlm-3b-8k-base"

    @property
    def name(self) -> str: 
        return "DPOV1"

    def __init__(self, device: str):
        self.device = device
        super().__init__() 
        self.penalty = 1.2  # Same penalty as the original [paper](https://arxiv.org/pdf/1909.05858.pdf).
        self.tokenizer = AutoTokenizer.from_pretrained(
            DirectPreferenceRewardModelV1.reward_model_name
        )
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLM.from_pretrained(
            DirectPreferenceRewardModelV1.reward_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))


    def reward_single(
        self, prompt: str, completion: str, name: str, with_penalty=True
    ) -> float:
        r"""Calculates a direct preference optimization (DPO) style reward for a completion,
        which is a reference model's average log-probability for completion tokens given a prompt.
        Uses guidance from https://github.com/eric-mitchell/direct-preference-optimization/blob/main/trainers.py.
        """
        with torch.no_grad():
            # Check if completion is
            if completion.strip() == "" or len(completion) <= 5:
                return -11  # exp(-11)=1.67e-5 < 2e-5=1/50257 (typical vocab size)

            # Tokenize the combined prompt + completion.
            combined = (
                self.tokenizer(prompt + completion, return_tensors="pt")
                .input_ids[0]
                .to(self.device)
            )  # [seq_len]
            # Tokenize only the prompt, to help determine prompt token length.
            prompt_part = (
                self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
            )  # [prompt_len]

            # Completion doesn't fit into model sequence, so return lowest reward.
            if self.tokenizer.model_max_length <= len(prompt_part):
                return -11.0  # exp(-11)=1.67e-5 < 2e-5=1/50257 (typical vocab size)

            # Truncate combined to fit into model max sequence length.
            if self.tokenizer.model_max_length < len(combined):
                combined = combined[: self.tokenizer.model_max_length]

            labels = combined.clone()  # [seq_len]
            # Ignore prompt part for calculating reward.
            labels[: len(prompt_part)] = -100
            # Label only each next token prediction ground-truth.
            labels = labels[1:]  # [seq_len-1]
            loss_mask = labels != -100  # [seq_len-1]
            # Dummy token to allow for indexing, but loss will be ignored.
            labels[labels == -100] = 0
            # Reshape for gather operation.
            labels = labels.unsqueeze(0).unsqueeze(2)  # [batch_size=1, seq_len-1, :]

            # Forward pass to calculate logit predictions for each sequence position.
            logits = self.model(
                combined.unsqueeze(0)
            ).logits  # [batch_size=1, seq_len, vocab_len]
            # Predict only where labels are available.
            logits = logits[:, :-1, :]  # [batch_size=1, seq_len-1, vocab_len]

            if with_penalty:
                # Apply penalty for repeated generation
                for i in range(len(prompt_part) + 1, len(combined) - 1):
                    logit = logits[:, i, :].clone()
                    inputs = combined[len(prompt_part) : i].clone()
                    logits[:, i, :] = self.logit_penalty(input_ids=inputs, logit=logit)

            # Rescale via log(softmax(logits)).
            logits = logits.log_softmax(-1)
            # Calculate the model's log-probability for each actual completion token.
            per_token_logps = torch.gather(logits, dim=2, index=labels).squeeze(
                2
            )  # [batch_size=1, seq_len-1]
            # Average log-probability over completion sequence.
            reward = (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(
                -1
            )  # [batch_size=1]
            reward = reward[0].cpu().detach()

            # NaNs can possibly arise through log(0)=-inf, replace with suitably small logits.
            if torch.isnan(reward) or torch.isinf(reward):
                return -11.0  # exp(-11)=1.67e-5 < 2e-5=1/50257 (typical vocab size)
            print("\n[DirectPreferenceRewardModelV1 Debugging]")
            # print("Combined input_ids:", combined)
            # print("Prompt part:", prompt_part)
            # print("Labels:", labels)
            if logits is not None:
                print("Logits shape:", logits.shape)
            if per_token_logps is not None:
                print("Per token logps shape:", per_token_logps.shape)
            print("Reward:", reward)
            return reward.item()
    def get_rewards(
        self, prompt: str, completions: List[str], name: str
    ) -> torch.FloatTensor:
        rewards = torch.tensor(
            [
                self.reward_single(prompt, completion, name)
                for completion in completions
            ],
            dtype=torch.float32,
        ).to(self.device)
        return rewards

    def logit_penalty(
        self, input_ids: torch.LongTensor, logit: torch.FloatTensor
    ) -> torch.FloatTensor:
        # Counts the unique tokens within each generation
        uniques, counts = input_ids.unique(return_counts=True)
        score = torch.gather(logit, 1, uniques.unsqueeze(0))

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(
            score < 0,
            score * (self.penalty**counts),
            score / (self.penalty**counts),
        )

        logit.scatter_(1, uniques.unsqueeze(0), score.to(logit.dtype))
        return logit

class DirectPreferenceRewardModelV2(BaseRewardModel):
    reward_model_name = "cerebras/btlm-3b-8k-base"

    @property
    def name(self) -> str:
        return "DPOV2"

    def __init__(self, device: str):
        self.device = device
        super().__init__()
        self.mean, self.var = torch.tensor([-11.78]).to(self.device), torch.tensor([4.36]).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(DirectPreferenceRewardModelV2.reward_model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model = AutoModelForCausalLM.from_pretrained(DirectPreferenceRewardModelV2.reward_model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
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

class DirectPreferenceRewardModelV3(BaseRewardModel):
    reward_model_name: str = "cerebras/btlm-3b-8k-base"

    @property
    def name(self) -> str: 
        return "DPOV3"

    def __init__(self, device: str):
        self.device = device
        super().__init__()  
        self.mean, self.var = torch.tensor([-11.78]).to(self.device), torch.tensor([4.36]).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            DirectPreferenceRewardModelV3.reward_model_name
        )
        self.model = AutoModelForCausalLM.from_pretrained(DirectPreferenceRewardModelV3.reward_model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def prepare_input(self, prompt: str, completions: List[str]) -> (torch.Tensor, List[Optional[torch.Tensor]]):
        """Tokenizes and prepares the input tensors."""
        prompt_part = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
        input_ids_list = []

        for completion in completions:
            combined = self.tokenizer(prompt + completion, truncation=False, return_tensors="pt").input_ids[0].to(self.device)

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
            padded_input_ids = torch.stack([torch.cat([ids, torch.full((max_length - len(ids),), 0).to(self.device)], 0) for ids in valid_input_ids])
            logits = self.model(padded_input_ids).logits[:, :-1, :].log_softmax(-1)

            rewards = []
            for idx, combined in enumerate(input_ids_list):
                if combined is None:
                    rewards.append(-11.)
                    continue
                labels = combined[1:].unsqueeze(1)
                per_token_logps = torch.gather(logits[idx], dim=1, index=labels).squeeze(1)
                loss_mask = (combined[1:] != -100)
                reward = (per_token_logps * loss_mask).sum() / loss_mask.sum()
                if torch.isnan(reward) or torch.isinf(reward):
                    rewards.append(-11.)
                else:
                    rewards.append(reward.item())

            return torch.tensor(rewards, dtype=torch.float16, device=self.device)

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        prompt_part, input_ids_list = self.prepare_input(prompt, completions)
        return self.reward_batch(prompt_part, input_ids_list)

def test_model(model1, model2, model3, prompt: str, completions: List[str]):
    try:
        v1_rewards = model1.get_rewards(prompt, completions, "dummy_name")
    except TypeError:
        v1_rewards = model1.get_rewards(prompt, completions)

    v2_rewards = model2.get_rewards(prompt, completions)
    v3_rewards = model3.get_rewards(prompt, completions)

    print(f"v1 Rewards: {v1_rewards}")
    print(f"v2 Rewards: {v2_rewards}")
    print(f"v3 Rewards: {v3_rewards}")
    return
    # difference = torch.abs(old_rewards - new_rewards)
    # print(f"Difference between old and new rewards: {difference}")

    # return difference

prompt = '''
In proposing more money now from Maryland, Virginia and the District, Mayer said, the governor was expanding on his earlier strategy.\n\n[Maryland to get $900-million federal full funding agreement for Purple Line]\n\nRep. Gerald E. Connolly (D-Va.) welcomed Hogan's change of mind and said he believed it came in response to the backlash to Hogan's position at the summit\n\nSummarize the preceding context in 4 sentences. Do not try to create questions or answers for your summarization.\n\n

'''
completions = [

"This offer represents a significant shift from Hogan's previously stated position at a recent regional summit where he refused to consider adding funds beyond those currently contributed annually by Maryland.",
'''
I don't know the answer to that question
'''
]

def load_model_v1(device):
    global model_v1
    model_v1 = DirectPreferenceRewardModelV1(device=device)

def load_model_v2(device):
    global model_v2
    model_v2 = DirectPreferenceRewardModelV2(device=device)

def load_model_v3(device):
    global model_v3
    model_v3 = DirectPreferenceRewardModelV3(device=device)

def main():
    # 1. Load models
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 4, "At least two GPUs are required for this setup."
    devices = [f"cuda:{i}" for i in range(num_gpus)]

    # Start threads
    thread1 = threading.Thread(target=load_model_v1, args=(devices[0],))
    thread2 = threading.Thread(target=load_model_v2, args=(devices[1],))
    thread3 = threading.Thread(target=load_model_v3, args=(devices[2],))
    thread1.start()
    thread2.start()
    thread3.start()
    thread1.join()
    thread2.join()
    thread3.join()

    test_model(model_v1, model_v2, model_v3, prompt, completions)
    # 2. Test tokenization
    tokenized_prompt_v1, tokenized_completions_v1 = model_v1.tokenizer(prompt), [model_v1.tokenizer(comp) for comp in completions]
    tokenized_prompt_v2, tokenized_completions_v2 = model_v2.tokenizer(prompt), [model_v2.tokenizer(comp) for comp in completions]
    
    assert tokenized_prompt_v1 == tokenized_prompt_v2, "Tokenized prompts differ."
    for idx, (comp_v1, comp_v2) in enumerate(zip(tokenized_completions_v1, tokenized_completions_v2)):
        assert comp_v1 == comp_v2, f"Tokenized completion {idx} differs."

    # 3. Test model inference
    input_text = "The quick brown fox"
    with torch.no_grad():
        inputs_v1 = model_v1.tokenizer(input_text, return_tensors="pt").input_ids.to(devices[0])
        logits_v1 = model_v1.model(inputs_v1).logits
        inputs_v2 = model_v2.tokenizer(input_text, return_tensors="pt").input_ids.to(devices[1])
        logits_v2 = model_v2.model(inputs_v2).logits

    assert torch.allclose(logits_v1, logits_v2, atol=1e-6), "Model logits differ."

    # 4. Test logits and probabilities for completions
    logits_list_v1 = [model_v1.model(model_v1.tokenizer(prompt + comp, return_tensors="pt").input_ids.to(devices[0])).logits for comp in completions]
    logits_list_v2 = [model_v2.model(model_v2.tokenizer(prompt + comp, return_tensors="pt").input_ids.to(devices[1])).logits for comp in completions]
    for idx, (logits_v1, logits_v2) in enumerate(zip(logits_list_v1, logits_list_v2)):
        assert torch.allclose(logits_v1, logits_v2, atol=1e-6), f"Logits for completion {idx} differ."

    # 5. Test high-level methods
    rewards_v1 = [model_v1.reward_single(prompt, comp, "TestName") for comp in completions]
    rewards_v2 = model_v2.get_rewards(prompt, completions)
    for idx, (reward_v1, reward_v2) in enumerate(zip(rewards_v1, rewards_v2)):
        assert torch.allclose(reward_v1, reward_v2, atol=1e-6), f"Rewards for completion {idx} differ."

    print("All tests passed!")

main()