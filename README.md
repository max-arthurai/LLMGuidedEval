# LLMGuidedEval

This is where I am proposing we centralize our work for the AIIA talk. Then after the talk, when we want to centralize our LLM guided eval research across the team more broadly we can make a copy of this repo owned by arthurai instead of me

# Action items:

DONE: 

- LLM response generation

- extract the actual rephrased text from the results dataframes and get rid of the "Here is my rephrased text" preamble, for example:

  * Claude:	Here is my attempt at rewriting the text with new language while preserving the meaning and information
 
  * LLaMa2: Here is my attempt at rewriting the text with new language while preserving the meaning and information



TODO:

For each task and each model, we have 31x3 = 93 evaluations and 110 A/B tests. So lets say ~200 evaluations.

We have 2 tasks and 5 models, which means we have a total of ~2000 evaluations to conduct.

If an evaluation takes ~20 seconds, we have ~40,000 seconds ~= 11 hours of computation to do.
  
- use each LLM to provide written feedback for every response

  * **31 responses**

- use each LLM to provide a binary (good/bad) feedback for every response

  * **31 responses**

- use each LLM to provide a letter grade (A/B/C/D/F) for every response

  * **31 responses**

- use each LLM to A/B test every pair of responses

  * to reduce the size of this experiment, lets maybe drop the triplets of each generation so that we have 11 choose 2 = 55 pairs instead of 31 choose 2 = 465 pairs to A/B test
 
  * for robustness, lets run each A/B test twice, swapping each time the order in which we place the A and B options.
   
  * So 55 pairs x 2 tests per pair = **110 A/B tests** total will need to get done for each inference

# Models

gpt-3.5-turbo & gpt4 (OpenAI), claude-2 (Anthropic), command (Cohere), LLaMa2 (Meta)

Each model was configured to generate with 256* maximum new tokens and a temperature of 0.7 (* I accidentally ran LLaMa2 with 500 maximum new tokens, but the average generations came out similar to the other models so I did not re-run them yet.)

# Data

## Summarization

`get_news.py` will use an API key I have registered with NewsAPI to fetch some articles. I chose 5 of these articles and wrote a ground truth summary for each one. I gave each LLM 3 attempts to summarize each article. Then, I gave each LLM 3 attempts at re-writing the ground truth summary that I wrote.

## Retrieval-Augmented Question-Answering

I wrote 5 climate-related questions and spent time reading the [IPCC 2023 climate change report](https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_SPM.pdf) (which also has a [shorter version](https://www.ipcc.ch/report/ar6/syr/resources/spm-headline-statements/) and a [longer version](https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf)) to understand the answers and select relevant text in order to mock what a retriever might return. I gave each LLM 3 attempts at answering each question with the mocked retrieval included in the prompt template. Then, I gave each LLM 3 attempts at re-writing the ground truth answer that I wrote (often using excerpts and phrases from the shorter IPCC report itself) 
