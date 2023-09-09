import concurrent.futures
from flask import Flask
import queue
import csv
import json
import time
import requests


app = Flask(__name__)

completion_endpoints = [
    "http://api1.url",
    "http://api2.url",
    # all local model endpoints go here
]

reward_endpoints = [
    "http://90.84.239.86:40357","http://90.84.239.86:40264","http://90.84.239.86:40332","http://90.84.239.86:40378", # Server 1, 4x4090, ssh -p 40243 root@90.84.239.86 -L 8080:localhost:8080
    "http://36.225.152.8:40594","http://36.225.152.8:40565","http://36.225.152.8:40512","http://36.225.152.8:40554", # Server 2, 4x4090, ssh -p 40543 root@36.225.152.8 -L 8080:localhost:8080
    "http://81.79.125.89:45654","http://81.79.125.89:45829","http://81.79.125.89:45395","http://81.79.125.89:45550", # Server 3, 4x4090 ssh -p 45648 root@81.79.125.89 -L 8080:localhost:8080
    ]

reward_servers = queue.Queue()
for url in reward_endpoints:
    reward_servers.put(url)

score_queue = queue.Queue()

fieldnames = ['Question', 'Completion Endpoint', 'Time Elapsed', 'Score']

def write_to_csv(data):
    with open('results.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow(data)

def request_to_completion_endpoint(completion_url, prompt):
    start_time = time.time()
    # Send the request to the completion endpoint
    # Return the received completion
    end_time = time.time()
    time_elapsed = end_time - start_time
    return (completion_response, time_elapsed)

def request_to_reward_endpoint(reward_url, completion_response):

    verify_token = "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n"
    data = {
        "verify_token": verify_token
        "prompt": prompt
        "responses": [completion_response]
    }
    response = requests.post(url, json=data)
    
    return response

def score_queued_completions():
    while True:
        # Constantly check if there are queued completions and available reward endpoints
        try:
            prompt, completion_url, completion_response, time_elapsed = score_queue.get_nowait()
            reward_url = reward_servers.get_nowait()
            
            score = request_to_reward_endpoint(reward_url, completion_response)
            # Write to CSV
            write_to_csv({
                'Question': prompt,
                'Completion Endpoint': completion_url,
                'Time Elapsed': time_elapsed,
                'Score': score
            })
            
            reward_servers.put(reward_url)
        except queue.Empty:
            continue

def process_completion_endpoint(completion_url, prompts):
    for prompt in prompts:
        completion_response, time_elapsed = request_to_completion_endpoint(completion_url, prompt)

        try:
            reward_url = reward_servers.get_nowait()
            score = request_to_reward_endpoint(reward_url, completion_response)
            write_to_csv({
                'Question': prompt,
                'Completion Endpoint': completion_url,
                'Time Elapsed': time_elapsed,
                'Score': score
            })
            reward_servers.put(reward_url)
        except queue.Empty:
            score_queue.put((prompt, completion_url, completion_response, time_elapsed))

if __name__ == '__main__':
    # Load prompts
    with open('dataset_enrichment/dataset/only_prompts.json', 'r') as f:
        prompts = json.load(f)

    # Process each completion endpoint
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(process_completion_endpoint, completion_endpoints, [prompts]*len(completion_endpoints)))
    
    # Start a thread to constantly check the score queue
    score_thread = threading.Thread(target=score_queued_completions)
    score_thread.start()

    app.run(debug=True)
