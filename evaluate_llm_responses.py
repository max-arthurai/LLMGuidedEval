import os
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

# define evaluation prompts
tasks = {
    "summary" : {
        "written" : PromptTemplate.from_template(
            "You are giving feedback on the quality of a summary."
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}\n=\n"
            "Give feedback on the strengths and/or weaknesses of the summary."
            "\n=Feedback=\n"), 
        "binary" : PromptTemplate.from_template(
            "You are giving a label based on the quality of a summary."
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}\n=\n"
            "Give a one-word quality label for the summary, either 'good' or 'bad'. Only respond with 'good' or 'bad', nothing else."
            "\n=Quality label=\n"), 
        "integer" : PromptTemplate.from_template(
            "You are giving a score based on the quality of a summary."
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}\n=\n"
            "Give a score 1-10 to this summary. 1 means irrelevant, 4 means no terrible errors but missing a basic thing or two, 6 means nothing wrong but not great, 8 means great, 10 means no possible improvements."
            "Only respond with the score, nothing else."
            "\n=Score=\n"), 
        "lettergrade" : PromptTemplate.from_template(
            "You are giving a grade based on the quality of a summary."
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}\n=\n"
            "Give a letter grade (A+ through F) to this summary. F means irrelevant, C means no terrible errors but missing a basic thing or two, B means nothing wrong, A means fantastic, A+ means no possible improvements."
            "Only respond with a letter grade, nothing else."
            "\n=Grade=\n"), 
        "abtest" : PromptTemplate.from_template(
            "You are choosing between two summaries based on how well they summarize an article."
            "\n=Article=\n{input_text}\n=Attempted Summary 0=\n{output_text_0}\n=Attempted Summary 1=\n{output_text_1}\n=\n"
            "Choose the better summary, Summary 0 or Summary 1. Only respond with '0' or '1', nothing else. "
            "\n=Choice=\n"), 
    },
    "qa" : {
        "written" : PromptTemplate.from_template(
            "You are giving feedback on the quality of an answer to a question."
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}\n=\n"
            "Give feedback on the strengths and/or weaknesses of the answer."
            "\n=Feedback=\n"), 
        "binary" : PromptTemplate.from_template(
            "You are giving a label based on the quality of an answer to a question."
            "=Context=\n{context}\n\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}\n=\n"
            "Give a one-word quality label (either 'good' or 'bad') to this answer. Only respond with 'good' or 'bad', nothing else."
            "\n=Quality label=\n"), 
        "integer" : PromptTemplate.from_template(
            "You are giving a score based on the quality of an answer to a question."
            "=Context=\n{context}\n\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}\n=\n"
            "Give a score 1-10 to this answer. 1 means irrelevant, 4 means no terrible errors but missing a basic thing or two, 6 means nothing wrong but not great, 8 means great, 10 means no possible improvements."
            "Only respond with the score, nothing else."
            "\n=Score=\n"), 
        "lettergrade" : PromptTemplate.from_template(
            "You are giving a grade based on the quality of an answer to a question."
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}\n=\n"
            "Give a letter grade (A+ through F) to this answer. F means irrelevant, C means no terrible errors but missing a basic thing or two, B means nothing wrong, A means fantastic, A+ means no possible improvements."
            "Only respond with a letter grade A, B, C, D, or F, nothing else."
            "\n=Grade=\n"), 
        "abtest" : PromptTemplate.from_template(
            "You are choosing between two answers based on how well they answer a question."
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Attempted Answer 0=\n{output_text_0}\n=Attempted Answer 1=\n{output_text_1}\n="
            "Choose the better answer, Answer 0 or Answer 0. Only respond with '0' or '1', nothing else. "
            "\n=Choice=\n"), 
    }
}

# define evaluator chains
# this evaluator chains dictionary is structured 
# task (e.g. summary) -> feedback type (e.g. lettergrade) -> model name -> chain (either LLMChain or LLaMa2Chain)
evaluator_chains = {} 
for task in tasks:
    evaluator_chains[task] = {}
    for prompt in tasks[task]:
        evaluator_chains[task][prompt] = {}
        for m in ["gpt35", "gpt4", "claude", "command"]:
            evaluator_chains[task][prompt][m] = LLMChain(llm=models[m], prompt=tasks[task][prompt])
        evaluator_chains[task][prompt]["llama2"] = LLaMa2Chain(llama2, tasks[task][prompt])

# load existing response data
summary_df = pd.read_csv("data/summarization/aug2023_news_summarization.csv")
articles = summary_df.article.values.tolist()
ground_truth_summaries = summary_df.ground_truth_summary.values.tolist()
qa_df = pd.read_csv("data/rag_qa/aug2023_ipcc_qa.csv")
questions = qa_df.question.values.tolist()
mocked_retrievals = qa_df.mocked_retrieval.values.tolist()
ground_truth_answers = qa_df.ground_truth_answer.values.tolist()
dfs = {"summary" : summary_df, "qa" : qa_df}
inputs = {"summary" : articles, "qa" : questions}

# define generic evaluation function for all feedback types
def evaluate(
    task="summary", # summary, qa
    feedback="binary", # written, binary, integer, lettergrade, abtest
    evaluator="gpt35", # gpt35, gpt4, claude, command, llama2
    candidate_response=None, # {gpt35, gpt4, claude, command, llama2} x {response, rephrase_gt} x {0, 1, 2}
    candidate_response_A=None, # {gpt35, gpt4, claude, command, llama2} x {response, rephrase_gt}
    candidate_response_B=None, # {gpt35, gpt4, claude, command, llama2} x {response, rephrase_gt}
    data_num=0, 
):
    if candidate_response:
        assert candidate_response_A is None and candidate_response_B is None and feedback != "abtest"
    if candidate_response_A is not None or candidate_response_B is not None:
        assert candidate_response is None and candidate_response_A is not None and candidate_response_B is not None and feedback=="abtest"

    # prepare inputs for the evaluator chain
    evaluator_input_dict = {
        "input_text" : inputs[task][data_num]
    }
    if task == "qa":
        evaluator_input_dict["context"] = mocked_retrievals[data_num]
    if feedback == "abtest":
        evaluator_input_dict["output_text_0"] = dfs[task][f"{candidate_response_A}"].values[data_num]
        evaluator_input_dict["output_text_1"] = dfs[task][f"{candidate_response_B}"].values[data_num]
    else:
        evaluator_input_dict["output_text"] = dfs[task][f"{candidate_response}"].values[data_num]
    
    # run evaluator chain
    evaluation = evaluator_chains[task][feedback][evaluator](evaluator_input_dict)["text"]
    return evaluation

def make_llm_evaluation_dataset_from_scratch():
    if not os.path.isdir("data/summarization/summary_evaluations"):
        os.mkdir("data/summarization/summary_evaluations")
    if not os.path.isdir("data/rag_qa/answer_evaluations"):
        os.mkdir("data/rag_qa/answer_evaluations")

    n_data = 5
    n_attempts = 3

    candidate_output_columns = {
        "summary" : ["ground_truth_summary"]+[f"{m}_{g}_{i}" for i in range(n_attempts) for g in ["summary", "rephrase_gt"] for m in models],
        "qa" : ["ground_truth_answer"]+[f"{m}_{g}_{i}" for i in range(n_attempts) for g in ["answer", "rephrase_gt"] for m in models],
    }
    evaluation_directories = {"summary" : "summarization/summary_evaluations", "qa" : "rag_qa/answer_evaluations"}

    for evaluator in models:
        for task in tasks:
            for data_num in range(n_data):

                # direct feedback
                for feedback in ["written", "binary", "integer", "lettergrade"]:
                    for candidate in candidate_output_columns[task]:
                        filename = f"data/{evaluation_directories[task]}/{evaluator}-{feedback}-{candidate}-{data_num}.txt"
                        if not os.path.exists(filename):
                            print(f"{evaluator} {task} data#{data_num} {feedback} {candidate}")
                            t0 = time()
                            direct_evaluation = evaluate(
                                task=task, 
                                feedback=feedback, 
                                evaluator=evaluator, 
                                candidate_response=candidate,
                                data_num=data_num
                            )
                            with open(filename, "w") as f:
                                f.write(direct_evaluation)
                            print("time", time() - t0)


                # A/B testing feedback 
                for candidate_A in candidate_output_columns[task]:
                    other_models = [m for m in candidate_output_columns[task] if m != candidate_A]
                    for candidate_B in other_models:
                        filename = f"data/{evaluation_directories[task]}/{evaluator}-abtest-{candidate_A}-{candidate_B}-{data_num}.txt"
                        # only consider candidates from attempt # 0 to narrow the space for now
                        if not os.path.exists(filename) and "_1" not in candidate_A and "_1" not in candidate_B and "_2" not in candidate_A and "_2" not in candidate_B:
                            print(f"{evaluator} {task} data#{data_num} abtest {candidate_A} {candidate_B}")
                            t0 = time()
                            ab_evaluation = evaluate(
                                task=task, 
                                feedback="abtest", 
                                evaluator=evaluator, 
                                candidate_response_A=candidate_A,
                                candidate_response_B=candidate_B,
                                data_num=data_num
                            )
                            with open(filename, "w") as f:
                                f.write(ab_evaluation)
                            print("time", time() - t0)

make_llm_evaluation_dataset_from_scratch()
