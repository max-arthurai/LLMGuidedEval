import os
import pandas as pd
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
import replicate
from time import time

# define models
gpt35 = ChatOpenAI(temperature=0.7, max_tokens=50)
gpt4 = ChatOpenAI(model='gpt-4', temperature=0.7, max_tokens=50)
claude = ChatAnthropic(temperature=0.7, max_tokens_to_sample=50)
command = Cohere(temperature=0.7, max_tokens=50)
class LlaMa2:
    """Callable LLaMa2 using replicate"""
    def __init__(self, model_name: str):
        self.model_name = model_name
    def __call__(self, input_text: str):
        replicate_output_generator = replicate.run(
            f"replicate/{self.model_name}",
            input={"prompt": input_text, "temperature" : 0.7, "max_new_tokens" : 50}
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
targets = ["gpt35", "gpt4", "claude", "command", "llama2"]
models = {
    "gpt35" : gpt35,
    "gpt4" : gpt4,
    "claude" : claude,
    "command" : command,
    "llama2" : llama2,
}

# import evaluation prompts
from calibrate_evaluator_prompts import tasks, optimized_task_evaluator_instructions

# define evaluators
# this evaluator chains dictionary is structured 
# task (e.g. summary) -> feedback type (e.g. lettergrade) -> model name -> chain (either LLMChain or LLaMa2Chain)
evaluators = {} 
for task in tasks:
    evaluators[task] = {}
    for prompt in tasks[task]:
        evaluators[task][prompt] = {}
        for m in ["gpt35", "gpt4", "claude", "command"]:
            evaluators[task][prompt][m] = LLMChain(llm=models[m], prompt=tasks[task][prompt])
        evaluators[task][prompt]["llama2"] = LLaMa2Chain(llama2, tasks[task][prompt])

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
    feedback="integer", # written, integer, lettergrade, abtest
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

    # prepare inputs for the evaluator
    evaluator_input_dict = {
        "input_text" : inputs[task][data_num], 
        "instruction" : optimized_task_evaluator_instructions[evaluator][task][feedback]
    }
    if task == "qa":
        evaluator_input_dict["context"] = mocked_retrievals[data_num]
    if feedback == "abtest":
        evaluator_input_dict["output_text_0"] = dfs[task][f"{candidate_response_A}"].values[data_num]
        evaluator_input_dict["output_text_1"] = dfs[task][f"{candidate_response_B}"].values[data_num]
    else:
        evaluator_input_dict["output_text"] = dfs[task][f"{candidate_response}"].values[data_num]
    
    # run evaluator
    evaluation = evaluators[task][feedback][evaluator](evaluator_input_dict)["text"]
    return evaluation


def make_llm_evaluation_dataset_from_scratch():
    if not os.path.isdir("data/summarization/summary_evaluations"):
        os.mkdir("data/summarization/summary_evaluations")
    if not os.path.isdir("data/rag_qa/answer_evaluations"):
        os.mkdir("data/rag_qa/answer_evaluations")

    n_data = 5
    n_attempts = 3

    candidate_output_columns = {
        "summary" : ["ground_truth_summary"]+[f"{m}_{g}_{i}" for i in range(n_attempts) for g in ["summary", "rephrase_gt"] for m in targets],
        "qa" : ["ground_truth_answer"]+[f"{m}_{g}_{i}" for i in range(n_attempts) for g in ["answer", "rephrase_gt"] for m in targets],
    }
    evaluation_directories = {"summary" : "summarization/summary_evaluations", "qa" : "rag_qa/answer_evaluations"}

    for evaluator in ["command"]:
        for task in tasks:
            for data_num in range(n_data):

                # direct feedback
                for feedback in ["integer", "lettergrade"]:#["written", "integer", "lettergrade"]:
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
                # for candidate_A in candidate_output_columns[task]:
                #     other_models = [m for m in candidate_output_columns[task] if m != candidate_A]
                #     for candidate_B in other_models:
                #         filename = f"data/{evaluation_directories[task]}/{evaluator}-abtest-{candidate_A}-{candidate_B}-{data_num}.txt"
                #         # only consider candidates from attempt # 0 to narrow the space for now
                #         if not os.path.exists(filename) and "_1" not in candidate_A and "_1" not in candidate_B and "_2" not in candidate_A and "_2" not in candidate_B:
                #             print(f"{evaluator} {task} data#{data_num} abtest {candidate_A} {candidate_B}")
                #             t0 = time()
                #             ab_evaluation = evaluate(
                #                 task=task, 
                #                 feedback="abtest", 
                #                 evaluator=evaluator, 
                #                 candidate_response_A=candidate_A,
                #                 candidate_response_B=candidate_B,
                #                 data_num=data_num
                #             )
                #             with open(filename, "w") as f:
                #                 f.write(ab_evaluation)
                #             print("time", time() - t0)

make_llm_evaluation_dataset_from_scratch()
