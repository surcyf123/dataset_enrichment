from flask import Flask, request, jsonify
import requests
import concurrent.futures

app = Flask(__name__)

# Define the 4 endpoints to which the prompt will be sent for completion
COMPLETION_ENDPOINTS = [
        "http://localhost:7776/generate",
        "http://localhost:7777/generate",
        "http://localhost:7778/generate",
        "http://localhost:7779/generate",
]

# Define the reward endpoint
reward_endpoint = "http://37.27.2.44:60181"

@app.route('/process_question', methods=['POST'])
def process_question():
    data = request.get_json()

    if 'prompt' not in data:
        return jsonify({"error": "Missing 'prompt' in request data."}), 400

    data = request.json
    prompt = data['prompt']
    completions_needed = data['completions_needed']
    max_time = data['max_time_allowed']
    question_value = data["question_value"]

    # Use threads to make parallel requests for completions
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(send_request, endpoint, {"prompt": prompt}) for endpoint in COMPLETION_ENDPOINTS]
        completions = [future.result() for future in concurrent.futures.as_completed(futures)]
    # Now score each completion
    scored_responses = []
    for completion in completions:
        if "response" in completion:
            scores = get_scores_from_reward_model(prompt, completion["response"])
            total_reward = scores[0]["Total Reward"]
            scored_responses.append({
                "completion": completion["response"],
                "score": total_reward,
                "model" :  completion["model"]
            })
    print(scored_responses)
    scored_responses.sort(key=lambda x: x['score'], reverse=True)

    return jsonify({"responses": scored_responses[:completions_needed]})

def send_request(endpoint, data):
    """Helper function to send request to the endpoint with the given data."""
    try:
        response = requests.post(endpoint, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Endpoint {endpoint} returned {response.status_code}"}
    except requests.RequestException as e:
        return {"error": str(e)}

def mock_send_request(endpoint, data):
    return {"response" : "pee"}

def get_scores_from_reward_model(original_prompt, response):
    '''Take the prompt, as well as the response, and return scores'''
    data = {
        "verify_token": "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n",
        "prompt": original_prompt,
        "completions": [response]
    }
    reward_response = requests.post(reward_endpoint, json=data)
    print(reward_response.json())
    if reward_response.status_code == 200:
        return reward_response.json()
    else:
        return {"error": f"Failed to get data: {reward_response.status_code}"}

if __name__ == '__main__':
    app.run(debug=True,port=8082,host='0.0.0.0')