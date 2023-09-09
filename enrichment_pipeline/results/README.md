The results will be placed into this folder

The columns are as follows:

```python
'prompt_index': i,
'num_tokens' : num_tokens,
'temperature': temperature,
'top_p': top_p,
'top_k': top_k,
'repetition_penalty': repetition_penalty,
'duration': duration,
'bert' : reward_scores[0]["Bert"][0],
'bert_norm' : reward_scores[0]["Bert"][1],
'dpo' : reward_scores[0]["DPO"][0],
'dpo_norm' : reward_scores[0]["DPO"][1],
'mpnet' : reward_scores[0]["MPNet"][0],
'mpnet_norm' : reward_scores[0]["MPNet"][1],
'rlhf' : reward_scores[0]["RLHF"][0],
'rlhf_norm' : reward_scores[0]["RLHF"][1],
'reciprocate' : reward_scores[0]["Reciprocate"][0],
'reciprocate_norm' : reward_scores[0]["Reciprocate"][1],
'total_reward' : reward_scores[0]["Total Reward"],
'prompt' : prompt,
'generated_text' : generated_text
```