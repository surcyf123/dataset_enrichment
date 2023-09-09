# Data Enrichment

## High Level Goals

- The goal of this is to download and setup recently made (<40 days), SOTA, quantized models into a flask endpoint for high-speed local inference.

- Prompt engineer and optimize hyperparameters of these models for local inference

- Test and evaluate the effects of the changes of various hyperparameter and prompt engineering changes on each reward model.

- Record results of tests, and show how certain changes in the model configuration affected the average change of the final reward score

- Use data from tests to develop a strategy for implementing the highest scoring local models, with specific configurations

- Create a backend with flask that can receive text inputs and direct the request to the best 2-3 models for that question type and return those answers

- Create an ensemble style approach to consistently score very high with every prompt in the sample dataset

- Automate everything as possible, log all results, and present findings after each day of work

## Breakdown of Tasks
- VastAI API for automatic deployment and testing
- Upload the results to Github
- TBC...

### Reward Endpoints
There are several endpoints, and we aim to have more on the way in order to increase the throughput of testing.

```
"http://213.173.102.136:10400"
"http://213.173.102.136:10401"
"http://213.173.102.136:10402"
```

### Connecting to Instance using the right key
I have registered vast.AI with the machine key so we can all use it

Example Command:
`ssh -o "IdentitiesOnly=yes" -i /home/bird/dataset_enrichment/credentials/autovastai -p 24364 root@ssh4.vast.ai -L 8080:localhost:8080`

### Models To Test
Below a is a list of quantized model strings I will use for testing. These are broken up into groups of 8 because most VAST.ai instances only have a maximum of 8x GPUs
```python
models_to_test = ['TheBloke/Pygmalion-2-13B-GPTQ','TheBloke/13B-Thorns-L2-GPTQ','TheBloke/Kimiko-13B-GPTQ','TheBloke/OpenBuddy-Llama2-13B-v11.1-GPTQ','TheBloke/Kimiko-13B-GPTQ',]
```