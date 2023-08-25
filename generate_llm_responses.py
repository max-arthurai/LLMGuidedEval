import numpy as np
import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
import replicate
from time import time

# define models
gpt35 = ChatOpenAI(temperature=0.7, max_tokens=256)
gpt4 = ChatOpenAI(model='gpt-4', temperature=0.7, max_tokens=256)
claude = ChatAnthropic(temperature=0.7, max_tokens_to_sample=256)
command = Cohere(temperature=0.7, max_tokens=256)
class LlaMa2:
    """Callable LLaMa2 using replicate"""
    def __init__(self, model_name: str):
        self.model_name = model_name
    def __call__(self, input_text: str):
        replicate_output_generator = replicate.run(
            f"replicate/{self.model_name}",
            input={"prompt": input_text, "temperature" : 0.7, "max_new_tokens" : 256}
        )
        replicate_output = "".join([x for x in replicate_output_generator])
        return replicate_output
llama2 = LlaMa2("llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1")
class LLaMa2Chain:
    """Callable class to mimic the langchain LLMChain callable syntax"""
    def __init__(self, model: LlaMa2, prompt : PromptTemplate):
        self.model = model
        self.prompt = prompt
    def __call__(self, input_dict):
        filled_template = self.prompt.format(**input_dict)
        output = self.model(filled_template)
        return {"text" : output}
models = {
    "gpt35" : gpt35,
    "gpt4" : gpt4,
    "claude" : claude,
    "command" : command,
    "llama2" : llama2,
}

# define prompts
prompts = {
    "summary" : PromptTemplate.from_template("Summarize the following news article: {article}\nSummary:"), 
    "qa" : PromptTemplate.from_template("Use the context to answer the question. Context: {context}\nQuestion: {question}\nAnswer:"), 
    "rephrase" : PromptTemplate.from_template("Rewrite this text using new language wherever possible while preserving the meaning and information: {text}\nRewritten:")
}

# define chains
# this chains dictionary is structured task prompt (e.g. summary) -> model name -> chain (either LLMChain or LLaMa2Chain)
chains = {} 
for p in prompts:
    chains[p] = {}
    for m in ["gpt35", "gpt4", "claude", "command"]:
        chains[p][m] = LLMChain(llm=models[m], prompt=prompts[p])
    chains[p]["llama2"] = LLaMa2Chain(llama2, prompts[p])

def load_summarization_data(n_data):
    articles = []
    ground_truth_summaries = []
    for n in range(n_data):
        with open(f"data/summarization/2023-08-23-articles/article_{n}.txt") as f:
            articles.append(f.read())
        with open(f"data/summarization/ground_truth_summaries/ground_truth_summary_{n}.txt") as f:
            ground_truth_summaries.append(f.read())
    return articles, ground_truth_summaries

def load_question_answering_data(n_data):
    questions = []
    mocked_retrievals = []
    ground_truth_answers = []
    for n in range(n_data):
        with open(f"data/rag_qa/questions/question_{n}.txt") as f:
            questions.append(f.read())
        with open(f"data/rag_qa/mocked_retrieval/mocked_retrieval_{n}.txt") as f:
            mocked_retrievals.append(f.read())
        with open(f"data/rag_qa/ground_truth_answers/ground_truth_answer_{n}.txt") as f:
            ground_truth_answers.append(f.read())
    return questions, mocked_retrievals, ground_truth_answers

def empty_response_dictionary(n_data, n_attempts):
    # each results dictionary is structured 
    # model name -> attempt # (e.g. 1st try, 2nd try, ...) -> data index (e.g. 1st article, 2nd article, ...)
    return {m : [["" for n in range(n_data)] for i in range(n_attempts)] for m in models}

def save_summarization_data(articles, ground_truth_summaries, summaries, rephrased_gt_summaries, n_attempts):
    # save summarization data
    summary_df = pd.DataFrame({})
    summary_df["article"] = articles
    summary_df["ground_truth_summary"] = ground_truth_summaries
    for m in models:
        for i in range(n_attempts):
            summary_df[f"{m}_summary_{i}"] = summaries[m][i]
            summary_df[f"{m}_rephrase_gt_{i}"] = rephrased_gt_summaries[m][i]
    summary_df.to_csv("data/summarization/aug2023_news_summarization.csv")

def save_question_answering_data(questions, mocked_retrievals, ground_truth_answers, answers, rephrased_gt_answers, n_attempts):
    # save question-answering data
    qa_df = pd.DataFrame({})
    qa_df["question"] = questions
    qa_df["mocked_retrieval"] = mocked_retrievals
    qa_df["ground_truth_answer"] = ground_truth_answers
    for m in models:
        for i in range(n_attempts):
            qa_df[f"{m}_answer_{i}"] = answers[m][i]
            qa_df[f"{m}_rephrase_gt_{i}"] = rephrased_gt_answers[m][i]
    qa_df.to_csv("data/rag_qa/aug2023_ipcc_qa.csv")

def make_llm_response_dataset_from_scratch(n_data = 5, n_attempts = 3):

    articles, ground_truth_summaries = load_summarization_data(n_data)
    questions, mocked_retrievals, ground_truth_answers = load_question_answering_data(n_data)
    summaries = empty_response_dictionary(n_data, n_attempts)
    rephrased_gt_summaries = empty_response_dictionary(n_data, n_attempts)
    answers = empty_response_dictionary(n_data, n_attempts)
    rephrased_gt_answers = empty_response_dictionary(n_data, n_attempts)
    
    # iterate over all models, attempts, and datapoints
    for m in models:
        for i in range(n_attempts):
            for n in range(n_data):

                print("summary")

                print(f"{m} generating {i}th summary on data {n}")
                t0 = time()
                summaries[m][i][n] = chains["summary"][m]({"article" : articles[n]})["text"]
                print("time", time() - t0)

                print(f"{m} generating {i}th gt summary rephrase on data {n}")
                t0 = time()
                rephrased_gt_summaries[m][i][n] = chains["rephrase"][m]({"text" : ground_truth_summaries[n]})["text"]
                print("time", time() - t0)

                save_summarization_data(articles, ground_truth_summaries, summaries, rephrased_gt_summaries, n_attempts)
                
                print("QA")
                
                print(f"{m} generating {i}th answer on data {n}")
                t0 = time()
                answers[m][i][n] = chains["qa"][m]({"question" : questions[n], "context" : mocked_retrievals[n]})["text"]
                print("time", time() - t0)

                print(f"{m} generating {i}th gt answer rephrase on data {n}")
                t0 = time()
                rephrased_gt_answers[m][i][n] = chains["rephrase"][m]({"text" : ground_truth_answers[n]})["text"]
                print("time", time() - t0)

                save_question_answering_data(questions, mocked_retrievals, ground_truth_answers, answers, rephrased_gt_answers, n_attempts)

