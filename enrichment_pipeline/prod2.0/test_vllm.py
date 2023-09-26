from vllm import LLM, SamplingParams

# Define the input prompt and sampling parameters
prompt = ["How are you?"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize the vLLM engine with your locally installed AWQ model
llm = LLM(model="/root/vicuna-7b-v1.5-awq", quantization="awq")

# Generate the output
outputs = llm.generate(prompt, sampling_params)

# Print the output
for output in outputs:
    prompt_text = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt_text!r}, Generated text: {generated_text!r}")


# import openai

# # Modify OpenAI's API key and API base to use vLLM's API server.
# openai.api_key = "EMPTY"
# openai.api_base = "http://localhost:8000/v1"

# # List models API
# models = openai.Model.list()
# print("Models:", models)

# model = models["data"][0]["id"]

# # Completion API
# stream = False
# completion = openai.Completion.create(
#     model=model,
#     prompt="A robot may not injure a human being",
#     echo=False,
#     n=2,
#     stream=stream,
#     logprobs=3)

# print("Completion results:")
# if stream:
#     for c in completion:
#         print(c)
# else:
#     print(completion)

# import requests

# # Define the API endpoint
# API_ENDPOINT = "http://localhost:8000/v1/completions"

# # Define the random question
# question = "What are the implications of quantum computing for cybersecurity?"

# # Set up the data payload for the request
# data = {
#     "model": "mosaicml/mpt-7b",
#     "prompt": question,
#     "n": 5,
#     "temperature": 0.8,
#     "top_p": 0.95
# }

# # Send the request

# response = requests.post(API_ENDPOINT, json=data)
# print(response.json())

# # Extract the completions from the response
# completions = response.json()['choices']

# # Print the completions
# for i, completion in enumerate(completions, 1):
#     print(f"Completion {i}: {completion['text']}")

