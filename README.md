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