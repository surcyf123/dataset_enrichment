from vllm import LLM, SamplingParams

# Define the input prompt and sampling parameters
prompt = ["How are you?"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Initialize the vLLM engine with your locally installed AWQ model
llm = LLM(model="/root/vicuna-7b-v1.5-awq")

# Generate the output
outputs = llm.generate(prompt, sampling_params)

# Print the output
for output in outputs:
    prompt_text = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt_text!r}, Generated text: {generated_text!r}")
