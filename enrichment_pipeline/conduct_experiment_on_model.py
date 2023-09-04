# %%
model_name = "TheBloke/Asclepius-13B-GPTQ"
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
    response = requests.post("http://localhost:7777/generate", json=data)
    elapsed_time = time.time() - start_time
    return response.json()['text'],elapsed_time

def get_scores_from_reward_model(original_prompt:str,response:str) -> Dict:
    '''Take the prompt, as well as the response, and return scores'''
    url = "http://213.173.102.136:10400"

    # Data to send
    data = {
        "verify_token": "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n",  # Your authentication token
        "prompt": original_prompt,
        "responses": [response]
    }

    # Make the POST request
    reward_response = requests.post(url, json=data)
    if reward_response.status_code == 200:
        return reward_response.json()
    else:
        print(f"Failed to get data: {reward_response.status_code}")
    

# Initialize CSV file and writer
with open(f'results/{hyperparameter_searches["num_tokens"]}-{model_name.replace("/","-")}.csv', mode='x', newline='') as csv_file:
    
    fieldnames = ['prompt_index', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'duration',
                  'reciprocate_reward', 'relevance_filter', 'rlhf_reward', 'combined_reward','prompt','generated_text']
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header to the CSV file
    csv_writer.writeheader()
    
    # Loop through the prompts and hyperparameters
    for i, prompt in tqdm(enumerate(prompts)):
        for num_tokens in hyperparameter_searches["num_tokens"]:
            for temperature in hyperparameter_searches["temperature"]:
                for top_p in hyperparameter_searches["top_p"]:
                    for top_k in hyperparameter_searches["top_k"]:
                        for repetition_penalty in hyperparameter_searches["repetition_penalty"]:
                            
                            generated_text, duration = call_model_with_params(prompt, temperature, top_p, top_k, repetition_penalty)
                            reward_scores = get_scores_from_reward_model(prompt, generated_text)
                            
                            reciprocate_reward = reward_scores["reward_details"]["reciprocate_reward_model"][0]
                            relevance_filter = reward_scores["reward_details"]["relevance_filter"][0]
                            rlhf_reward = reward_scores["reward_details"]["rlhf_reward_model"][0]
                            combined_reward = reward_scores["rewards"][0]
                            
                            # Write a row to the CSV file
                            csv_writer.writerow({
                                'prompt_index': i,
                                'num_tokens' : num_tokens,
                                'temperature': temperature,
                                'top_p': top_p,
                                'top_k': top_k,
                                'repetition_penalty': repetition_penalty,
                                'duration': duration,
                                'reciprocate_reward': reciprocate_reward,
                                'relevance_filter': relevance_filter,
                                'rlhf_reward': rlhf_reward,
                                'combined_reward': combined_reward,
                                'prompt' : prompt,
                                'generated_text' : generated_text
                            })
                    
                    
    
