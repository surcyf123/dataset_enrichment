# %%
import sys
model_name = sys.argv[1]
local_port = sys.argv[2]
experiment_id = sys.argv[3]
reward_endpoint = sys.argv[4]
gpu_name = sys.argv[5]

if len(sys.argv > 6):
    prompt_formatting_found = True
    prompt_template = sys.argv[6]
    print("Prompt Template:")
    print(prompt_template)

elif len(sys.argv == 5):
    prompt_formatting_found = True
    try:
        with open(model_name+"/README.md",'r') as readmefile:
            content = readmefile.read()
        prompt_template = content.split("<!-- prompt-template start -->")[1].split("<!-- prompt-template end -->")[0].split("```")[1].rstrip().lstrip()
    except:
        print("No valid prompt formatting found.")
        prompt_formatting_found = False

# %%

import requests
import json
import time
from typing import Tuple, Dict
import csv
import random
with open("../dataset/only_prompts.json", "r") as f:
    prompts = json.load(f)
random.seed(42)
random.shuffle(prompts)
sample_size = 250
sampled_prompts = prompts[:sample_size]
from tqdm import tqdm





hyperparameter_searches = {
    "num_tokens" : [200],
    "temperature" : [0.7],
    "top_p" : [0.7],
    "top_k" : [50],
    "repetition_penalty" : [1.0]
}

def call_model_with_params(prompt:str,num_tokens:int,temperature:float, top_p:float, top_k:int, repetition_penalty:float, prompt_formatting:bool) -> Tuple[str,float]:
    '''Returns the generated text, along with how long it took to execute'''
    if prompt_formatting:
        data = {
        "prompt": prompt_template.replace("{prompt}",prompt.rstrip()),
        "max_new_tokens": num_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
        "stopwords": []
    }
    else:
        data = {
        "prompt": prompt,
        "max_new_tokens": num_tokens,
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
    
    for attempt in range(10): #makes 10 attempts to reward model
        try:
            # do thing
            reward_response = requests.post(url, json=data)
            if reward_response.status_code == 200:
                return reward_response.json()
            else:
                raise Exception("It wasnt a success")

        except:
            print(f"Failed to connect to reward model, attempt: {str(attempt)}")
            time.sleep(1)
        
        break

    
    
    
        
    
# %%

# -------------------------------- with formatting ---------------------------#
if prompt_formatting_found:
    for num_tokens in hyperparameter_searches["num_tokens"]:
        with open(f'results/{num_tokens}-{model_name}-fmt.csv', mode='a', newline='') as csv_file:
            fieldnames = ['prompt_index','num_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'duration','bert','bert_norm','dpo','dpo_norm','mpnet','mpnet_norm','rlhf','rlhf_norm','reciprocate','reciprocate_norm','total_reward','prompt','generated_text']
            csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            csv_writer.writeheader()


    print("Experiment Starting")
    # Loop through the prompts and hyperparameters
    for i, prompt in tqdm(enumerate(sampled_prompts)):
        for num_tokens in hyperparameter_searches["num_tokens"]:
            for temperature in hyperparameter_searches["temperature"]:
                for top_p in hyperparameter_searches["top_p"]:
                    for top_k in hyperparameter_searches["top_k"]:
                        for repetition_penalty in hyperparameter_searches["repetition_penalty"]:
                            with open(f'results/{num_tokens}-{model_name}-fmt.csv', mode='a', newline='') as csv_file:
                                retries = 0
                                while True:
                                    try:
                                        generated_text, duration = call_model_with_params(prompt,num_tokens, temperature, top_p, top_k, repetition_penalty,prompt_formatting=True)
                                        reward_scores = get_scores_from_reward_model(prompt, generated_text)
                                        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
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
                                    except Exception as e:
                                        retries += 1
                                        if retries > 5:
                                            print("Number of Retries exceeded limit, skipping row.")
                                            break
                                        print(f"{retries} - Failed to Call Model(s):")
                                        print(e)
                                    break



    # Writing the stats
    import pandas as pd
    for num_tokens in hyperparameter_searches["num_tokens"]:       
        with open(f'results/{num_tokens}-{model_name}-fmt.txt', 'w') as f:
            df = pd.read_csv(f'results/{num_tokens}-{model_name}-fmt.csv')
            f.write(f'gpu_name {gpu_name}\n')
            
            mean_duration = df['duration'].mean()
            f.write(f'mean_duration {mean_duration}\n')
            total_relevance_pass_rate = (df[df['total_reward'] != 0].shape[0])/len(df)
            f.write(f'relevance_pass_rate {total_relevance_pass_rate}\n')

            #bert
            pass_mean_bert = df['bert'].mean()
            f.write(f'bert_raw_avg {pass_mean_bert}\n')
            pass_mean_bert_norm = df['bert_norm'].mean()
            f.write(f'bert_pass_rate {pass_mean_bert_norm}\n')

            #mpnet
            pass_mean_dpo = df['mpnet'].mean()
            f.write(f'mpnet_raw_avg {pass_mean_dpo}\n')
            pass_mean_dpo_norm = df['mpnet_norm'].mean()
            f.write(f'mpnet_pass_rate {pass_mean_dpo_norm}\n')

            #dpo
            pass_mean_dpo = df['dpo'].mean()
            f.write(f'dpo_mean {pass_mean_dpo}\n')
            pass_mean_dpo_norm = df['dpo_norm'].mean()
            f.write(f'dpo_norm_mean {pass_mean_dpo_norm}\n')


            # rlhf
            pass_mean_rlhf = df['rlhf'].mean()
            f.write(f'rlhf_mean {pass_mean_rlhf}\n')
            pass_mean_rlhf_norm = df['rlhf_norm'].mean()
            f.write(f'rlhf_mean_norm {pass_mean_rlhf_norm}\n')

            #reciprocate
            pass_mean_reciprocate_reward = df[df['total_reward'] != 0]['reciprocate'].mean()
            f.write(f'reciprocate_reward_mean {pass_mean_reciprocate_reward}\n')
            pass_mean_reciprocate_reward_norm = df[df['total_reward'] != 0]['reciprocate_norm'].mean()
            f.write(f'reciprocate_reward_mean_norm {pass_mean_reciprocate_reward}\n')                      




# Initialize CSV file and writer    

# Write the header to the CSV file
for num_tokens in hyperparameter_searches["num_tokens"]:
    with open(f'results/{num_tokens}-{model_name}.csv', mode='a', newline='') as csv_file:
        fieldnames = ['prompt_index','num_tokens', 'temperature', 'top_p', 'top_k', 'repetition_penalty', 'duration','bert','bert_norm','dpo','dpo_norm','mpnet','mpnet_norm','rlhf','rlhf_norm','reciprocate','reciprocate_norm','total_reward','prompt','generated_text']
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()


print("Experiment Starting")
# Loop through the prompts and hyperparameters
for i, prompt in tqdm(enumerate(sampled_prompts)):
    for num_tokens in hyperparameter_searches["num_tokens"]:
        for temperature in hyperparameter_searches["temperature"]:
            for top_p in hyperparameter_searches["top_p"]:
                for top_k in hyperparameter_searches["top_k"]:
                    for repetition_penalty in hyperparameter_searches["repetition_penalty"]:
                        with open(f'results/{num_tokens}-{model_name}.csv', mode='a', newline='') as csv_file:
                            retries = 0
                            while True:
                                try:
                                    generated_text, duration = call_model_with_params(prompt,num_tokens, temperature, top_p, top_k, repetition_penalty, prompt_formatting = False)
                                    reward_scores = get_scores_from_reward_model(prompt, generated_text)
                                    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
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
                                except Exception as e:
                                    retries += 1
                                    if retries > 5:
                                        print("Number of Retries exceeded limit, skipping row.")
                                        break
                                    print(f"{retries} - Failed to Call Model(s):")
                                    print(e)
                                break

# Writing the stats
import pandas as pd
for num_tokens in hyperparameter_searches["num_tokens"]:       
    with open(f'results/{num_tokens}-{model_name}.txt', 'w') as f:
        df = pd.read_csv(f'results/{num_tokens}-{model_name}.csv')
        f.write(f'gpu_name {gpu_name}\n')
        
        mean_duration = df['duration'].mean()
        f.write(f'mean_duration {mean_duration}\n')
        total_relevance_pass_rate = (df[df['total_reward'] != 0].shape[0])/len(df)
        f.write(f'relevance_pass_rate {total_relevance_pass_rate}\n')

        #bert
        pass_mean_bert = df['bert'].mean()
        f.write(f'bert_raw_avg {pass_mean_bert}\n')
        pass_mean_bert_norm = df['bert_norm'].mean()
        f.write(f'bert_pass_rate {pass_mean_bert_norm}\n')

        #mpnet
        pass_mean_dpo = df['mpnet'].mean()
        f.write(f'mpnet_raw_avg {pass_mean_dpo}\n')
        pass_mean_dpo_norm = df['mpnet_norm'].mean()
        f.write(f'mpnet_pass_rate {pass_mean_dpo_norm}\n')

        #dpo
        pass_mean_dpo = df['dpo'].mean()
        f.write(f'dpo_mean {pass_mean_dpo}\n')
        pass_mean_dpo_norm = df['dpo_norm'].mean()
        f.write(f'dpo_norm_mean {pass_mean_dpo_norm}\n')


        # rlhf
        pass_mean_rlhf = df['rlhf'].mean()
        f.write(f'rlhf_mean {pass_mean_rlhf}\n')
        pass_mean_rlhf_norm = df['rlhf_norm'].mean()
        f.write(f'rlhf_mean_norm {pass_mean_rlhf_norm}\n')

        #reciprocate
        pass_mean_reciprocate_reward = df[df['total_reward'] != 0]['reciprocate'].mean()
        f.write(f'reciprocate_reward_mean {pass_mean_reciprocate_reward}\n')
        pass_mean_reciprocate_reward_norm = df[df['total_reward'] != 0]['reciprocate_norm'].mean()
        f.write(f'reciprocate_reward_mean_norm {pass_mean_reciprocate_reward}\n')                        
    

  
    
    
with open(f'/root/ckpts/{experiment_id}_ckpt4', 'w') as fp:
    pass