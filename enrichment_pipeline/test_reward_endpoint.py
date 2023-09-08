import requests



# The API endpoint
url = "http://127.0.0.1:8008/"



# Your data
data = {
"verify_token": "SjhSXuEmZoW#%SD@#nAsd123bash#$%&@n",
"prompt": "Which city is the capital of France?",
"completions": [
"Paris is the capital of France.",
"London is the capital of England.",
"Berlin is the capital of Germany."
]
}


# response = requests.post(url, json=data)
print(response.text)
print(response.status_code)

if response.text:
    print(response.json())
else:
    print("Empty response from server")

print(response.json())