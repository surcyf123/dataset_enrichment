from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Tuple
from abc import abstractmethod

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
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # assert self.tokenizer.pad_token is not None, f"Tokenizer's pad token not set for {self.name}!"
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
            inputs = self.tokenizer(prompts, completions, padding=True, truncation=True, max_length=512, return_tensors='pt').to(self.device)
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
        self.model = AutoModelForCausalLM.from_pretrained(
            DirectPreferenceRewardModelV1.reward_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(self.device)

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
        bt.logging.trace(f"DirectPreferenceRewardModel | rewards: {rewards.tolist()}")
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
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # assert self.tokenizer.pad_token is not None, f"Tokenizer's pad token not set for {self.name}!"
        self.model = AutoModelForCausalLM.from_pretrained(DirectPreferenceRewardModelV2.reward_model_name, trust_remote_code=True, torch_dtype=torch.float16).to(self.device)

    def reward_single(self, prompt: str, completion: str) -> float:
        with torch.no_grad():
            combined = self.tokenizer(prompt + completion, return_tensors="pt").input_ids[0].to(self.device)
            prompt_part = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
            if len(prompt_part) >= self.tokenizer.model_max_length or len(combined) == 0:
                return -11.

            combined = combined[:self.tokenizer.model_max_length]
            labels = combined.clone()
            labels[:len(prompt_part)] = -100
            labels = labels[1:]

            logits = self.model(combined.unsqueeze(0)).logits[:, :-1, :]
            logits = logits.log_softmax(-1)
            per_token_logps = torch.gather(logits, dim=2, index=labels.unsqueeze(0).unsqueeze(2)).squeeze(2)
            mask = (labels != -100).unsqueeze(0)
            reward = (per_token_logps[mask]).mean()

            if torch.isnan(reward) or torch.isinf(reward):
                return -11.
            return reward.item()

    def reward_batch(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        with torch.no_grad():
            combined = self.tokenizer([prompt]*len(completions), completions, padding=True, truncation=True, return_tensors="pt").to(self.device)
            prompt_part = self.tokenizer(prompt, return_tensors="pt").input_ids[0].to(self.device)
            
            if len(prompt_part) >= self.tokenizer.model_max_length:
                return torch.full((len(completions),), -11).to(dtype=torch.float16)

            labels = combined.input_ids.clone()
            labels[:, :len(prompt_part)] = 0
            labels = labels[:, 1:]

            logits = self.model(**combined).logits[:, :-1, :]
            logits = logits.log_softmax(-1)
            per_token_logps = torch.gather(logits, dim=2, index=labels.unsqueeze(2)).squeeze(2)
            mask = (labels != 0)
            
            counts = mask.sum(dim=1)
            sums = (per_token_logps * mask).sum(dim=1)
            rewards = sums / counts.to(dtype=torch.float16)
            
            invalid_rewards = torch.isnan(rewards) | torch.isinf(rewards)
            rewards[invalid_rewards] = -11.0
            
            return rewards

    def get_rewards(self, prompt: str, completions: List[str]) -> torch.FloatTensor:
        return self.reward_batch(prompt, completions)


def test_model(model_v1, model_v2, prompt: str, completions: List[str]):
    rewards_v1 = model_v1.get_rewards(prompt, completions)
    rewards_v2 = model_v2.get_rewards(prompt, completions).to(rewards_v1.device)

    print(f"{model_v1.name} Rewards: {rewards_v1}")
    print(f"{model_v2.name} Rewards: {rewards_v2}")

    difference = torch.abs(rewards_v1 - rewards_v2)
    print(f"Difference between {model_v1.name} and {model_v2.name} rewards: {difference}")

    return difference

prompt = '''
Maryland Gov. Larry Hogan (R) upended the regional debate over Metro funding Monday by offering to give the transit system an extra $500 million over four years if Virginia, the District and the federal government each do the same.\n\nHogan's proposal, made in a letter delivered Monday morning to Virginia Gov. Terry McAuliffe (D) and D.C. Mayor Muriel E. Bowser (D), narrowed their differences over funding and appeared to increase chances that the region could agree on a plan to save the agency.\n\nBut it remained to be seen whether the other three parties \u2014 especially the federal government and Virginia \u2014 would go along. Some politicians grumbled that Hogan only made the proposal because he knew it was unlikely to be accepted, and a Metro board member predicted the federal government would balk.\n\nOverall, however, top Metro officials and other regional leaders praised Hogan for taking an important first step toward reaching consensus, while they warned that the plan falls short of a permanent solution.\n\nHogan's action marked a dramatic reversal from his position in a contentious, closed-door, regional summit two weeks ago. There, Hogan shocked McAuliffe and Bowser by saying Maryland would not give Metro any additional funds beyond what it already contributes each year.\n\n[Behind closed doors, region\u2019s leaders clashed sharply over Metro funding]\n\nOn Monday, Hogan reaffirmed the stance he took at the summit against new taxes to support Metro, and he complained that Maryland contributes more than its fair share to the struggling transit agency. But he took a new approach regarding more money.\n\n\"The needs of the Metro system are immediate and overwhelming,\" Hogan wrote in the detailed, four-page letter. \"Given the current crisis, the State of Maryland is prepared to invest an additional $500 million in increased Metro funding over the next four years if the Commonwealth of Virginia, the District of Columbia and the federal government all commit to do the same.\"\n\nHogan's about-face appeared prompted partly by intense criticism of his earlier opposition, officials said, both from other regional actors and from a strongly worded editorial in The Washington Post headlined, \"Larry Hogan to Metro: Drop Dead.\" Hogan is expected to seek reelection next year, and an anti-Metro stance could hurt him in vote-rich Montgomery and Prince George's counties.\n\nBut Hogan spokesman Doug Mayer suggested the governor had taken the adamant position at the summit as a bargaining ploy at the start of what he expected to be a prolonged process.\n\nHogan is \"always negotiating,\" Mayer said.\n\nMayer also rejected the idea that Hogan had altered his position, noting that the governor emphasized in the letter his previous stance that the federal government ought to contribute more to Metro. In proposing more money now from Maryland, Virginia and the District, Mayer said, the governor was expanding on his earlier strategy.\n\n[Maryland to get $900-million federal full funding agreement for Purple Line]\n\nRep. Gerald E. Connolly (D-Va.) welcomed Hogan's change of mind and said he believed it came in response to the backlash to Hogan's position at the summit\n\nSummarize the preceding context in 4 sentences. Do not try to create questions or answers for your summarization.\n\n

'''
completions = [

"Here is a summary of the given text in four sentences:\n\n1. Maryland Governor Larry Hogan has offered to provide an additional $500 million in funding over four years to help save the Metrorail system if Virginia, DC, and the federal government match this amount.\n2. This offer represents a significant shift from Hogan's previously stated position at a recent regional summit where he refused to consider adding funds beyond those currently contributed annually by Maryland.",
'''
I don't know the answer to that question
'''
]

def main():
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 3, "At least three GPUs are required for this setup."
    
    devices = [f"cuda:{i}" for i in range(num_gpus)]
    # Test ReciprocateRewardModel versions
    # print("\nTesting ReciprocateRewardModel versions...")
    # model_v1_reciprocate = ReciprocateRewardModelV1(device=devices[2])
    # model_v2_reciprocate = ReciprocateRewardModelV2(device=devices[3])
    # test_model(model_v1_reciprocate, model_v2_reciprocate, prompt, completions)

    # Test OpenAssistantRewardModel versions
    print("\nTesting OpenAssistantRewardModel versions...")
    model_v1_open_assistant = OpenAssistantRewardModelV1(device=devices[2])
    model_v2_open_assistant = OpenAssistantRewardModelV2(device=devices[3])
    test_model(model_v1_open_assistant, model_v2_open_assistant, prompt, completions)

    # Test DirectPreferenceRewardModel versions
    print("\nTesting DirectPreferenceRewardModel versions...")
    model_v1_direct_preference = DirectPreferenceRewardModelV1(device=devices[2])
    model_v2_direct_preference = DirectPreferenceRewardModelV2(device=devices[3])
    test_model(model_v1_direct_preference, model_v2_direct_preference, prompt, completions)

# Sample run for the revised script
main()



