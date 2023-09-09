# %%
import sys
model_name_or_path = sys.argv[1]
local_port = sys.argv[2]
experiment_id = sys.argv[3]
reward_endpoint = sys.argv[4]

import requests
import json
import time
from typing import Tuple, Dict
import csv
# %%
from tqdm import tqdm
with open("../dataset/only_prompts.json", "r") as f:
    prompts = json.load(f)
    

hyperparameter_searches = {
    "num_tokens" : [100,200,300,400,500],
    "temperature" : [0.7],
    "top_p" : [0.7],
    "top_k" : [None],
    "repetition_penalty" : [None]
}

def call_model_with_params(prompt:str,temperature:float, top_p:float, top_k:int, repetition_penalty:float) -> Tuple[str,float]:
    '''Returns the generated text, along with how long it took to execute'''
    data = {
    "prompt": prompt,
    "max_new_tokens": 300,
    "temperature": temperature,
    "top_p": top_p,
    "top_k": top_k,
    "repetition_penalty": repetition_penalty,
    "stopwords": []
}
    start_time = time.time()
    response = requests.post(f"http://localhost:{local_port}/generate", json=data)
    elapsed_time = time.time() - start_time
    return response.json()['text'],elapsed_time
# %%
def get_scores_from_reward_model(original_prompt:str,response:str) -> Dict:
    '''Take the prompt, as well as the response, and return scores'''
    url = reward_endpoint

    # Data to send
    data = {
        "verify_token": "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n",  # Your authentication token
        "prompt": original_prompt,
        "completions": [response]
    }

    # Make the POST request
    reward_response = requests.post(url, json=data)
    if reward_response.status_code == 200:
        return reward_response.json()
    else:
        print(f"Failed to get data: {reward_response.status_code}")
    
# %%
# Initialize CSV file and writer    
    # Write the header to the CSV file
    
print("Experiment Starting")
# Loop through the prompts and hyperparameters
for i, prompt in tqdm(enumerate(prompts)):
    for num_tokens in hyperparameter_searches["num_tokens"]:
        for temperature in hyperparameter_searches["temperature"]:
            for top_p in hyperparameter_searches["top_p"]:
                for top_k in hyperparameter_searches["top_k"]:
                    for repetition_penalty in hyperparameter_searches["repetition_penalty"]:
                        with open(f'results/{num_tokens}-{model_name_or_path.replace("TheBloke/","")}.csv', mode='a', newline='') as csv_file:
                            fieldnames = ['prompt_index','num_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'duration','bert','bert_norm','dpo','dpo_norm','mpnet','mpnet_norm','rlhf','rlhf_norm','reciprocate','reciprocate_norm','total_reward','prompt','generated_text']
                            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                            csv_writer.writeheader()
                            generated_text, duration = call_model_with_params(prompt, temperature, top_p, top_k, repetition_penalty)
                            reward_scores = get_scores_from_reward_model(prompt, generated_text)             
                            
                            try:
                                # Write a row to the CSV file
                                csv_writer.writerow({
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
                                })
                            except:
                                print("Failed to Index Reward Scores:")
                                print(reward_scores)
print("Experiment Complete")                    
                    
    
