from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
import os
import pandas as pd
import re
import replicate
from time import time

# define models
gpt35 = ChatOpenAI(temperature=0.7, max_tokens=10)
gpt4 = ChatOpenAI(model='gpt-4', temperature=0.7, max_tokens=256)
claude = ChatAnthropic(temperature=0.7, max_tokens_to_sample=10)
command = Cohere(temperature=0.7, max_tokens=10)
class LlaMa2:
    """Callable LLaMa2 using replicate"""
    def __init__(self, model_name: str):
        self.model_name = model_name
    def __call__(self, input_text: str):
        replicate_output_generator = replicate.run(
            f"replicate/{self.model_name}",
            input={"prompt": input_text, "temperature" : 0.7, "max_new_tokens" : 10}
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
            "\n=Instruction=\n{instruction}"
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}"
            "\n=Feedback=\n "), 
        "integer" : PromptTemplate.from_template(
            "You are giving a score based on the quality of a summary."
            "\n=Instruction=\n{instruction}"
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}"
            "\n=Score=\n "), 
        "lettergrade" : PromptTemplate.from_template(
            "You are giving a grade based on the quality of a summary."
            "\n=Instruction=\n{instruction}"
            "\n=Article=\n{input_text}\n=Summary=\n{output_text}"
            "\n=Grade=\n "), 
        "abtest" : PromptTemplate.from_template(
            "You are choosing between two summaries based on how well they summarize an article."
            "Choose the better summary on the basis of relevance, importance, and accuracy. Only respond with '0' or '1', nothing else."
            "\n=Article=\n{input_text}\n=Summary 0=\n{output_text_0}\n=Summary 1=\n{output_text_1}"
            "\n=Choice=\n "), 
    },
    "qa" : {
        "written" : PromptTemplate.from_template(
            "You are giving feedback on the correctness of an answer to a question."
            "\n=Instruction=\n{instruction}"
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}"
            "\n=Feedback=\n "), 
        "integer" : PromptTemplate.from_template(
            "You are giving a score based on the correctness of an answer to a question."
            "\n=Instruction=\n{instruction}"
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}"
            "\n=Score=\n "), 
        "lettergrade" : PromptTemplate.from_template(
            "You are giving a grade based on the correctness of an answer to a question."
            "\n=Instruction=\n{instruction}"
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Attempted Answer=\n{output_text}"
            "\n=Grade=\n "), 
        "abtest" : PromptTemplate.from_template(
            "You are choosing between two answers based on how well they answer a question."
            "\n=Instruction=\n{instruction}"
            "\n=Context=\n{context}\n=Question=\n{input_text}\n=Answer 0=\n{output_text_0}\n=Answer 1=\n{output_text_1}"
            "\n=Choice=\n "), 
    }
}
baseline_instructions = {
    "summary" : {
        "written" : "Give one sentence of feedback on the summary with respect to its relevance, importance, and accuracy. Be extremely strict and critical when it comes to relevance, importance, and accuracy.",
        "integer" : "Give a score 1-10 to this summary. 1 means irrelevant, 5 means errors, 10 means no possible improvements. Be extremely harsh, strict, and critical with respect to its relevance, importance, and accuracy. Only respond with the score, nothing else.",
        "lettergrade" : "Give a letter grade (A+ through F) to this summary. F means irrelevant, C means errors, A+ means no possible improvements. Be extremely harsh, strict, and critical with respect to its relevance, importance, and accuracy. Only respond with a letter grade, nothing else.",
        "abtest" : "Choose the better summary on the basis of relevance, importance, and accuracy. Only respond with '0' or '1', nothing else."
    },
    "qa" : {
        "written" : "Give one sentence of feedback on the answer with respect to its correctness. Be extremely strict and critical when it comes to correctness.",
        "integer" : "Give a score 1-10 to this answer. 1 means irrelevant, 5 means errors, 10 means no possible improvements. Be extremely harsh, strict, and critical with respect to correctness. Only respond with the score, nothing else.",
        "lettergrade" : "Give a letter grade (A+ through F) to this answer. F means irrelevant, C means errors, A+ means no possible improvements. Be extremely harsh, strict, and critical with respect to correctness. Only respond with a letter grade, nothing else.",
        "abtest" : "Choose the better answer on the basis of correctness. Only respond with '0' or '1', nothing else."
    }
}

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

# define sample inputs & outputs
sample_article = """
Title: ‘And Just Like That’ Renewed for Season 3 at Max

“And Just Like That” has been renewed for Season 3 at Max.

The news comes just ahead of the Season 2 finale, which is set to debut on Max on Aug. 24. The second season originally premiered on June 22.

“We are thrilled to spend more time in the ‘Sex And The City’ universe telling new stories about the lives of these relatable and aspirational characters played by these amazing actors,” said executive producer and showrunner Michael Patrick King. ‘And Just Like That…’ here comes season three.”

The series reunites Sarah Jessica Parker, Cynthia Nixon, and Kristin Davis as they reprise their roles from “Sex and the City.” The new season also saw the return of John Corbett as Aidan. The Season 2 cast also includes Sara Ramírez, Sarita Choudhury, Nicole Ari Parker, Karen Pittman, Mario Cantone, David Eigenberg, Evan Handler, Christopher Jackson, Niall Cunningham, Cathy Ang, and Alexa Swinton. Original series cast member Kim Cattrall is set to appear in a cameo in the series finale.

“We are delighted to share that since the launch of season two, ‘And Just Like That…’ ranks as the #1 Max Original overall, and is the most-watched returning Max Original to date,” said Sarah Aubrey, head of original content for Max. “As we approach the highly anticipated season finale on Thursday, we raise our cosmos to Michael Patrick King and his magnificent team of writers, producers, cast and crew who continue to charm us, 25 years later, with dynamic friendships and engaging stories. We cannot wait for audiences to see where season three will take our favorite New Yorkers.”

King developed the series in addition to serving as executive producer and showrunner. Parker, Davis, and Nixon also executive produce. John Melfi, Julie Rottenberg, and Elisa Zuritsky are also executive producers on the series. The HBO series “Sex and the City” was created by Darren Star and based on the book “Sex and the City” by Candace Bushnell."""

sample_perfect_summary = "And Just Like That has been renewed for a third season on Max ahead of its Season 2 finale. The show, a continuation of 'Sex and the City' sees Sarah Jessica Parker, Cynthia Nixon, and Kristin Davis returning to their roles. John Corbett also returned as Aidan in Season 2, and Kim Cattrall is set for a finale cameo. Since its second season, the series has become the top Max Original. The series is developed by Michael Patrick King and based on the HBO show and Candace Bushnell's book."
sample_ok_summary = "There is going to be a third season of And Just Like That the hit HBO Max Original series. The news comes just ahead of the Season 2 finale, which is set to debut on Max on Aug. 24. The show has become the most watched Max Original. The second season originally premiered on June 22. The series reunites Sarah Jessica Parker, Cynthia Nixon, and Kristin Davis as they reprise their roles from “Sex and the City.” The new season also saw the return of John Corbett as Aidan. King developed the series in addition to serving as executive producer and showrunner. Parker, Davis, and Nixon also executive produce. John Melfi, Julie Rottenberg, and Elisa Zuritsky are also executive producers on the series. This is a show on HBO Max and it is an Original show starring Sarah Jessica Parker reprising her role from 25 years ago as Carrie Bradshaw."
sample_awful_summary = "The hit show HBO Max is coming to Sarah Jessica Parker this fall, with season two returning to be thrilled at developing executive producers. There are people named John, Julia, Elisa, Sarah, Michael, Cynthia discussed. Through the power of television, their vision has come together to produce the series."
sample_irrelevant = "Most of the time, the sun rises in the morning and sets at night."

sample_question = "When are new And Just Like That episodes gonna come out?"
sample_perfect_answer = "The article doesn't mention a date for season 3, only that the show has been renewed for a third season. It does mention, though, that the season 2 finale airs August 24."
sample_ok_answer = "Doesn't say."
sample_awful_answer = "Aug. 24"



# define evaluation parsers

def get_number_grade(s: str) -> float:
    # This regex pattern will match numbers 1-10 with optional decimal values for numbers other than 10.
    pattern = r'\b(10|[1-9](?:\.\d)?)\b'
    match = re.search(pattern, s)
    return float(match.group()) if match else -1.0

def get_letter_grade(s: str) -> str:
    # This regex pattern will match letter grades A+ to F, with optional + or -.
    pattern = r'\b([A-E][+-]?|F)(?!\w)'
    match = re.search(pattern, s)
    return match.group() if match else "nan"

assert get_number_grade("My score is 7.5 out of 10.") == 7.5
assert get_number_grade("My score is 3 out of 10.") == 3
assert get_letter_grade("F") == "F"
assert get_letter_grade("I got a grade of B+ for the assignment.") == "B+"
assert get_letter_grade("A-\n my reasoning is") == "A-"

grade_parser = {"integer" : get_number_grade, "lettergrade" : get_letter_grade}

prompt_calibration_prompt = PromptTemplate.from_template(
    "You are re-writing an instruction for an LLM to get its evaluation to be calibrated to expected results. "
    "You are given the LLM's current instruction and the results of its calibration tests (as well as what the expected outputs should be). "
    "You will return the re-written instruction in order to get the LLM to output the expected results on these same inputs."
    "\n=Current instruction=\n{current_instruction}"
    "\n=Input text=\n{input_text}"
    "\n=Perfect response (expected evaluation: {expected_perfect}, current LLM evaluation: {evaluation_perfect})=\n{candidate_perfect}"
    "\n=OK response (expected evaluation: {expected_ok}, current LLM evaluation: {evaluation_ok})=\n{candidate_ok}"
    "\n=Awful response (expected evaluation: {expected_awful}, current LLM evaluation: {evaluation_awful})=\n{candidate_awful}"
    "\n=Irrelevant response (expected evaluation: {expected_irrelevant}, current LLM evaluation: {evaluation_irrelevant})=\n{candidate_irrelevant}"
    "\n=Your re-written instruction to calibrate the LLM evaluation to be the expected evaluation:=\n "
)

# these are the most calibrated prompt instructions observed yet
gpt35_summary_lettergrade_instruction = "Give a letter grade (A+ through F) to the following summary based on its relevance, importance, and accuracy. F means irrelevant, C indicates the presence of errors or misinformation, and A+ signifies that there are no possible improvements. Be strict in your judgement, but do not over-penalize minor errors. Only respond with a letter grade, nothing else."
claude_summary_lettergrade_instruction = "Assign a letter grade (A+ through F) to the provided summary based on its quality. An F grade signifies the summary is irrelevant or doesn't relate to the original text. A C grade indicates the summary has some errors or missing key points from the original text. An A+ grade means the summary accurately and succinctly encapsulates all the important information from the original text with no improvements possible. Be critical in your assessment, but ensure the grade reflects the summary's relevance to the original text, its accuracy, and the completeness of its content. Provide only the letter grade as your response."
command_summary_lettergrade_instruction = "Please provide an assessment of the given summary by assigning it a letter grade from A+ to F. Ensure your grading is calibrated as follows: A+ should only be given if the summary is fully accurate, relevant, and complete. There should be no errors, omissions, or flaws. Any weaknesses should lower the grade from there."
llama2_summary_lettergrade_instruction = baseline_instructions["summary"]["lettergrade"]

gpt35_summary_integer_instruction = baseline_instructions["summary"]["integer"]
claude_summary_integer_instruction = ""
command_summary_integer_instruction = ""
llama2_summary_integer_instruction = ""

gpt35_qa_lettergrade_instruction = ""
claude_qa_lettergrade_instruction = ""
command_qa_lettergrade_instruction = ""
llama2_qa_lettergrade_instruction = ""

gpt35_qa_integer_instruction = ""
claude_qa_integer_instruction = ""
command_qa_integer_instruction = ""
llama2_qa_integer_instruction = ""

sample_inputs = {
    "summary" : sample_article,
    "qa" : sample_question
}

expected_calibrated_results = {
    "lettergrade" : {"perfect" : "A+", "ok" : "B-", "awful" : "D", "irrelevant" : "F"},
    "integer" : {"perfect" : "10", "ok" : "6", "awful" : "3", "irrelevant" : "1"}
}

sample_outputs = {
    "summary" : {"perfect" : sample_perfect_summary, "ok" : sample_ok_summary, "awful" : sample_awful_summary, "irrelevant" : sample_irrelevant},
    "qa" : {"perfect" : sample_perfect_answer, "ok" : sample_ok_answer, "awful" : sample_awful_answer, "irrelevant" : sample_irrelevant}
}


def calibrate_evaluator(
    evaluator_name, 
    task,
    feedback,
    calibrator_name="gpt4", 
    baseline_instruction=None
):

    instruction = baseline_instruction
    if baseline_instruction is None:
        instruction = baseline_instructions[task][feedback]

    calibrator = LLMChain(llm=models[calibrator_name], prompt=prompt_calibration_prompt)

    while True:
        print("current instruction:", instruction)

        print("________________________________________")
        calibrator_input = {"current_instruction" : instruction, "input_text" : sample_inputs[task]}
        for candidate in ["perfect", "ok", "awful", "irrelevant"]:
            evaluator_input = {
                "instruction" : instruction,
                "input_text" : sample_inputs[task],
                "output_text" : sample_outputs[task][candidate]
            }
            if task == "qa":
                evaluator_input["context"] = sample_article
            evaluation = evaluators[task][feedback][evaluator_name](evaluator_input)["text"]
            parsed_evaluation = grade_parser[feedback](evaluation)
            print("unparsed", evaluation)
            print("parsed", parsed_evaluation)

            calibrator_input[f"candidate_{candidate}"] = sample_outputs[task][candidate]
            calibrator_input[f"expected_{candidate}"] = expected_calibrated_results[feedback][candidate]
            calibrator_input[f"evaluation_{candidate}"] = parsed_evaluation

        calibrated_instruction = calibrator(calibrator_input)["text"]   
        print(calibrated_instruction)
        user_edited_calibrated_instruction = input("----\nIf the new instruction needs to be parsed, type it now. Otherwise just hit enter:\n>>>")
        if user_edited_calibrated_instruction == "":
            instruction = calibrated_instruction
        else:
            instruction = user_edited_calibrated_instruction


calibrate_evaluator(
    "gpt35", 
    "qa",
    "integer",
    sample_context=sample_article
)