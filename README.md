# LLMGuidedEval

This is where I am proposing we centralize our work for the AIIA talk. Then after the talk, when we want to centralize our LLM guided eval research across the team more broadly we can make a copy of this repo owned by arthurai instead of me

# Models

gpt-3.5-turbo & gpt4 (OpenAI), claude-2 (Anthropic), command (Cohere), LLaMa2 (Meta)

Each model was configured to generate with 256* maximum new tokens and a temperature of 0.7 (* I accidentally ran LLaMa2 with 500 maximum new tokens, but the average generations came out similar to the other models so I did not re-run them yet.)

# Data

## Summarization

`get_news.py` will use an API key I have registered with NewsAPI to fetch some articles. I chose 5 of these articles and wrote a ground truth summary for each one. I gave each LLM 3 attempts to summarize each article. Then, I gave each LLM 3 attempts at re-writing the ground truth summary that I wrote.

## Retrieval-Augmented Question-Answering

I wrote 5 climate-related questions and spent time reading the [IPCC 2023 climate change report](https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_SPM.pdf) (which also has a [shorter version](https://www.ipcc.ch/report/ar6/syr/resources/spm-headline-statements/) and a [longer version](https://www.ipcc.ch/report/ar6/syr/downloads/report/IPCC_AR6_SYR_LongerReport.pdf)) to understand the answers and select relevant text in order to mock what a retriever might return. I gave each LLM 3 attempts at answering each question with the mocked retrieval included in the prompt template. Then, I gave each LLM 3 attempts at re-writing the ground truth answer that I wrote (often using excerpts and phrases from the shorter IPCC report itself) 
