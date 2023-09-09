from flask import Flask, request, jsonify
import threading
import queue
import requests
from datetime import datetime, timedelta
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


app = Flask(__name__)

completion_endpoints = [
    # Add local model urls here
    ]

reward_endpoints = [
    "http://90.84.239.86:40357","http://90.84.239.86:40264","http://90.84.239.86:40332","http://90.84.239.86:40378", # Server 1, 4x4090, ssh -p 40243 root@90.84.239.86 -L 8080:localhost:8080
    "http://36.225.152.8:40594","http://36.225.152.8:40565","http://36.225.152.8:40512","http://36.225.152.8:40554", # Server 2, 4x4090, ssh -p 40543 root@36.225.152.8 -L 8080:localhost:8080
    "http://81.79.125.89:45654","http://81.79.125.89:45829","http://81.79.125.89:45395","http://81.79.125.89:45550", # Server 3, 4x4090 ssh -p 45648 root@81.79.125.89 -L 8080:localhost:8080
    ]

completion_queue = queue.Queue()
for url in completion_endpoints:
    completion_queue.put(url)

reward_queue = queue.Queue()
for url in reward_endpoints:
    reward_queue.put(url)

datastore = {}
datastore_lock = threading.Lock()

completions_list = []
completions_lock = threading.Lock()


# without using completion endpoints, just a predefined_answer
@app.route('/process_question', methods=['POST'])
def process_question():
    data = request.json
    prompt = data['prompt']
    completions_needed = data['completions_needed']
    max_time = data['max_time_allowed']

    # Reset the completions list
    with completions_lock:
        completions_list.clear()

    # Generate predefined completions
    predefined_answer = "This is my answer. I think it is a good answer."
    completions_list.extend([predefined_answer] * completions_needed)

    for completion in completions_list:
        t = threading.Thread(target=score_completion, args=(prompt, completion))
        t.start()

    return jsonify({"completions_list": completions_list})


# with using the completion endpoints
# @app.route('/process_question', methods=['POST'])
# def process_question():
#     data = request.json
#     prompt = data['prompt']
#     completions_needed = data['completions_needed']
#     max_time = data['max_time_allowed']

#     logging.info(f"Received request for prompt: {prompt[:30]}... with {completions_needed} completions needed.")

#     # Reset the completions list
#     with completions_lock:
#         completions_list.clear()

#     # Send the task to worker threads
#     for _ in range(completions_needed):
#         t = threading.Thread(target=worker_thread, args=(prompt,))
#         t.start()

#     # Implement a timeout mechanism
#     start_time = datetime.now()
#     timeout = timedelta(seconds=max_time)
#     while len(completions_list) < completions_needed and datetime.now() - start_time < timeout:
#         pass

#     logging.info(f"Returning {len(completions_list)} completions.")
#     return jsonify({"completions_list": completions_list})

def worker_thread(prompt):
    logging.info("Worker thread started.")
    
    # Get a completion endpoint from the queue
    completion_url = completion_queue.get()

    logging.info(f"Requesting completion from: {completion_url}.")
    
    # Make the request to get the completion
    completion = request_to_completion_endpoint(completion_url, prompt)
    
    if completion:
        logging.info("Received completion. Preparing to score.")
        
        # If successful, send it for scoring
        t = threading.Thread(target=score_completion, args=(completion,))
        t.start()
    else:
        logging.warning("Completion was not successful.")
    
    # Put the endpoint back in the queue
    completion_queue.put(completion_url)

def request_to_completion_endpoint(completion_url, prompt):
    start_time = time.time()
    # Send the request to the completion endpoint
    # Return the received completion
    end_time = time.time()
    logging.info(f"Completion endpoint took {end_time - start_time} seconds")
    time_elapsed = end_time - start_time
    return completion

def score_completion(prompt, completion):
    logging.info("Preparing to score completion.")
    
    # Get a reward endpoint from the queue
    reward_url = reward_queue.get()

    verify_token = "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n"
    data = {
        "verify_token": verify_token,
        "prompt": prompt,
        "completions": [completion],
    }
    
    try:
        response = requests.post(reward_url, json=data)
        logging.info(f"Scored completion with response: {json.dumps(response.json())}")
        
        # Store the completion in the shared list
        with completions_lock:
            completions_list.append(completion)
    except Exception as e:
        logging.error(f"Error scoring completion: {e}")
    
    # Put the endpoint back in the queue
    reward_queue.put(reward_url)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=3000, debug=False)