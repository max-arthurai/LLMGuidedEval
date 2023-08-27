from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import Cohere
from langchain.prompts import PromptTemplate
import os
import pandas as pd
import re
import replicate
from time import time

# # define models
# gpt35 = ChatOpenAI(temperature=0.7, max_tokens=32)
# gpt4 = ChatOpenAI(model='gpt-4', temperature=0.7, max_tokens=256)
# claude = ChatAnthropic(temperature=0.7, max_tokens_to_sample=32)
# command = Cohere(temperature=0.7, max_tokens=32)
# class LlaMa2:
#     """Callable LLaMa2 using replicate"""
#     def __init__(self, model_name: str):
#         self.model_name = model_name
#     def __call__(self, input_text: str):
#         replicate_output_generator = replicate.run(
#             f"replicate/{self.model_name}",
#             input={"prompt": input_text, "temperature" : 0.7, "max_new_tokens" : 32}
#         )
#         replicate_output = "".join([x for x in replicate_output_generator])
#         return replicate_output
# llama2 = LlaMa2("llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1")
# class LLaMa2Chain:
#     """Callable class to mimic the langchain LLMChain callable syntax"""
#     def __init__(self, model: LlaMa2, prompt : PromptTemplate):
#         self.model = model
#         self.prompt = prompt
#     def __call__(self, input_dict):
#         filled_template = self.prompt.format(**input_dict)
#         output = self.model(filled_template)
#         return {"text" : output}
# models = {
#     "gpt35" : gpt35,
#     "gpt4" : gpt4,
#     "claude" : claude,
#     "command" : command,
#     "llama2" : llama2,
# }

# define task evaluation prompt templates
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

# define baseline instructions for task evaluation
baseline_instructions = {
    "summary" : {
        "written" : "Give one sentence of feedback on the summary with respect to its relevance, importance, and accuracy. Be extremely strict and critical when it comes to relevance, importance, and accuracy.",
        "integer" : "Give a score 0-10 to this summary. 0 means irrelevant, 5 means errors, 10 means no possible improvements. Be extremely harsh, strict, and critical with respect to its relevance, importance, and accuracy. ALWAYS start your feedback by specifying the score, followed by an explanation of why you have given this score. For instance, 'Score: 5. The summary is only partially correct.'",
        "lettergrade" : "Give a letter grade (A+ through F) to this summary. F means irrelevant, C means errors, A+ means no possible improvements. Be extremely harsh, strict, and critical with respect to its relevance, importance, and accuracy. ALWAYS start your feedback by specifying the grade, followed by an explanation of why you have given this grade. For instance, 'Grade: B-. The summary is only partially correct.'",
        "abtest" : "Choose the better summary on the basis of relevance, importance, and accuracy. Only respond with '0' or '1', nothing else."
    },
    "qa" : {
        "written" : "Give one sentence of feedback on the answer with respect to its correctness. Be extremely strict and critical when it comes to correctness.",
        "integer" : "Give a score 0-10 to this answer. 0 means irrelevant, 5 means errors, 10 means no possible improvements. Be extremely harsh, strict, and critical with respect to correctness. ALWAYS start your feedback by specifying the score, followed by an explanation of why you have given this score. For instance, 'Score: 5. The answer is only partially correct.'",
        "lettergrade" : "Give a letter grade (A+ through F) to this answer. F means irrelevant, C means errors, A+ means no possible improvements. Be extremely harsh, strict, and critical with respect to correctness. ALWAYS start your feedback by specifying the grade, followed by an explanation of why you have given this grade. For instance, 'Grade: B-. The answer is only partially correct.'",
        "abtest" : "Choose the better answer on the basis of correctness. Only respond with '0' or '1', nothing else."
    }
}

# define evaluators
# this evaluator chains dictionary is structured 
# task (e.g. summary) -> feedback type (e.g. lettergrade) -> model name -> chain (either LLMChain or LLaMa2Chain)
# evaluators = {} 
# for task in tasks:
#     evaluators[task] = {}
#     for prompt in tasks[task]:
#         evaluators[task][prompt] = {}
#         for m in ["gpt35", "gpt4", "claude", "command"]:
#             evaluators[task][prompt][m] = LLMChain(llm=models[m], prompt=tasks[task][prompt])
#         evaluators[task][prompt]["llama2"] = LLaMa2Chain(llama2, tasks[task][prompt])

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

sample_question = "When is season 3 of And Just Like That gonna come out?"
sample_perfect_answer = "The question is not answerable because the article doesn't mention a date for season 3."
sample_ok_answer = "I don't know."
sample_awful_answer = "Aug. 24"

sample_inputs = {
    "summary" : sample_article,
    "qa" : sample_question
}

sample_outputs = {
    "summary" : {
        "perfect" : sample_perfect_summary, 
        "ok" : sample_ok_summary, 
        "awful" : sample_awful_summary, 
        "irrelevant" : sample_irrelevant
    },
    "qa" : {
        "perfect" : sample_perfect_answer, 
        "ok" : sample_ok_answer, 
        "awful" : sample_awful_answer, 
        "irrelevant" : sample_irrelevant
    }
}

expected_calibrated_results = {
    "lettergrade" : {
        "perfect" : "A+", 
        "ok" : "B-", 
        "awful" : "D", 
        "irrelevant" : "F"
    },
    "integer" : {
        "perfect" : "10", 
        "ok" : "5", 
        "awful" : "3", 
        "irrelevant" : "0"
    }
}

# define evaluation parsers
def get_number_grade(s: str) -> float:
    # This regex pattern will match numbers 0-10 with optional decimal values for numbers other than 10.
    pattern = r'\b(10|[0-9](?:\.\d)?)\b'
    match = re.search(pattern, s)
    return float(match.group()) if match else -1.0

def get_letter_grade(s: str) -> str:
    # This regex pattern will match letter grades A+ to F, with optional + or -.
    pattern = r'\b([A-E][+-]?|F)(?!\w)'
    match = re.search(pattern, s)
    return match.group() if match else "nan"

assert get_number_grade("My score is 7.5 out of 10.") == 7.5
assert get_number_grade("My score is 3 out of 10.") == 3
assert get_number_grade("I would give it a 0 fr fr") == 0
assert get_letter_grade("F") == "F"
assert get_letter_grade("I got a grade of B+ for the assignment.") == "B+"
assert get_letter_grade("A-\n my reasoning is") == "A-"

grade_parser = {"integer" : get_number_grade, "lettergrade" : get_letter_grade}

prompt_calibration_prompt = PromptTemplate.from_template(
    "You are re-writing an instruction for an LLM to get its evaluation to be calibrated to expected results. "
    "You are given the LLM's current instruction and the results of its calibration tests (as well as what the expected outputs should be). "
    "Describe why the previous evaluation was wrong so that the new evaluation results are closer to the expected evaluation. "
    "Very weak LLMs need to be able to understand your instructions so be extremely precise."
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
claude_summary_integer_instruction = "Rate the summary on a scale of 0-10. A score of 0 indicates the summary is completely irrelevant to the original text, a score of 5 suggests multiple inaccuracies or misunderstandings, and a score of 10 represents a perfect summary with no room for improvement. Be rigorous and exacting in your evaluation, paying particular attention to the relevance, the factual accuracy, and the ability to effectively convey the main points of the original text. Please note that your response should only include the numerical score."
command_summary_integer_instruction = baseline_instructions["summary"]["integer"]
llama2_summary_integer_instruction = baseline_instructions["summary"]["integer"]

gpt35_qa_lettergrade_instruction = "Evaluate the answer by grading it from A+ through F. Give an A+ to answers that are perfectly correct or provide an explanation as to why the question cannot be answered with the given information. However, if the answer simply states 'I don't know' without providing any additional explanation or context, grade it as a B-. If the answer provides incorrect or misleading information, grade it as a D. Answers that do not pertain to the question at all, providing completely irrelevant information, should receive an F."
claude_qa_lettergrade_instruction = "Please give a letter grade (A+ through F) based on the correctness and helpfulness of the response. An 'A+' grade should be given to responses that accurately answer the question with the most relevant information, even if it is not the expected answer. A 'B-' grade is for responses that don't provide specific information but still address the question. A 'D' grade is for responses providing incorrect information while an 'F' is for completely irrelevant responses. Only mention the letter grade, like 'A', and nothing else."
command_qa_lettergrade_instruction = "Please assess the following responses based on their relevance, precision, and completeness in relation to the posed question. Award an A+ to answers that appropriately address the query, supplying pertinent data even if the exact information is not accessible. A response that might lack in detail but acknowledges the inability to provide the desired data should receive a B-. Grade with a D in cases where the response provides false or misleading information. An F should be assigned to responses that do not correspond to the query at all and offer unrelated information. Please note that even if the response does not provide the exact answer, as long as it correctly addresses the question and provides relevant information, it should be graded as A+. For example, if the question asks for a specific date and the response indicates that the date is not mentioned, it should still be graded as A+ as it provides relevant information. B- should be given to responses that acknowledge the inability to provide the requested information without any further details. Misleading or incorrect information, particularly false dates or facts that seem relevant but are actually incorrect, should be graded as D. Efforts to present irrelevant or completely unrelated information should receive an F grade."
llama2_qa_lettergrade_instruction = """Assign a letter grade (ranging from A+ to F) to the provided response. ALWAYS start your feedback by specifying the grade, followed by a thorough explanation of why you have given this grade. For instance, 'Grade: B. The reason behind this grade is that the response is partially correct.'\n\nTake into consideration the following criteria while grading: \n\n1. If the response accurately answers the question and provides relevant information, such as 'The article doesn't mention a date for season 3', it should be graded as an 'A+'. \n\n2. If the response does not provide the exact answer but still remains relevant to the question, for example, 'I don't know', it should be graded as a 'B-'.\n\n3. If the response includes incorrect information, such as providing a random date when the release date is not known, like 'Aug. 24', it should be graded as a 'D'. \n\n4. If the response is completely unrelated to the question, like 'Most of the time, the sun rises in the morning and sets at night', it should be graded as an 'F'. \n\nRemember to judge the response based on its relevance and correctness, not on the structure or style of the answer."""

gpt35_qa_integer_instruction = """Please rate the answer on a scale from 1 to 10 based on the following guidelines: \n\n- If the answer is correct and specifically addresses the question with information found within the given context, assign it a score of 10.\n\n- If the question cannot be answered due to lack of relevant information in the provided context, and the LLM correctly identifies this by stating that the context does not provide the answer, also give it a score of 10.\n\n- If the LLM response is evasive or vague such as "I don't know" without referring to the lack of information in the context, give it a score of 5.\n\n- If the information is present in the context but is inaccurately represented in the response, give this a score of 3.\n\n- Any responses that are completely unrelated to the question or the context should be assigned a score of 0."""
claude_qa_integer_instruction = "Provide a score ranging from 0 to 10 based on the accuracy and usefulness of the response. Accuracy refers to whether the information provided is true or false, while usefulness refers to how beneficial the information is to the user. ALWAYS initiate your response by stating your score and then explaining the reason behind your score. For example, 'Score: 5. The reason behind this score is that the response is only partially correct or helpful.' Remember, a perfect response that is both accurate and helpful should be scored a 10, an OK or partially correct/helpful response should be scored a 5, a wrong but somewhat related response should be scored a 3, and a completely irrelevant response should be scored as 0."
command_qa_integer_instruction = "Score the correctness of the answer on a scale of 0 to 10 where:\n\n 10 - the answer directly addresses the question asked and provides a factually accurate and relevant response. If the question asks for specifics that are not known, the answer acknowledges this and does not speculate.\n\n5 - the answer attempts to address the question but may not contain some degree of speculation, ambiguity, or factual errors. However, the core content is still somewhat relevant.\n\n3 - the answer is tangential.\n\n0 - the answer is irrelevant."
llama2_qa_integer_instruction = "You need to evaluate the provided response on a scale of 0 to 10. Begin your evaluation by presenting the score, then provide a detailed explanation for your scoring. Your explanation should focus on how precise, helpful, and relevant the response is to the question asked. For example, 'Score: 5. This score is given because the response, while not incorrect, does not provide a precise or helpful answer to the question.' Here are some guiding principles for your evaluation:\n\n1. When the response precisely answers the question with the information available, even if it is to confirm the unavailability of the answer, it should be scored as 10. \n2. An ambiguous response or a response that does not provide a direct answer to the question, but isn't incorrect should be scored as 5. \n3. A response that provides incorrect information should be scored as 3. This includes responses that provide specific information that is false or misleading. \n4. If the response is completely unrelated to the question, it should receive the lowest score of 0.\n\nRemember, these are only guidelines, there can be variations based on the specific context of the responses. It's important to consider the relevance and accuracy of the response in the context of the specific question being asked"

optimized_task_evaluator_instructions = {
    "gpt35" : {
        "summary" : {
            "integer" : gpt35_summary_integer_instruction,
            "lettergrade" : gpt35_summary_lettergrade_instruction
        },
        "qa" : {
            "integer" : gpt35_qa_integer_instruction,
            "lettergrade" : gpt35_qa_lettergrade_instruction
        }
    },
    "claude" : {
        "summary" : {
            "integer" : claude_summary_integer_instruction,
            "lettergrade" : claude_summary_lettergrade_instruction
        },
        "qa" : {
            "integer" : claude_qa_integer_instruction,
            "lettergrade" : claude_qa_lettergrade_instruction
        }
    },
    "command" : {
        "summary" : {
            "integer" : command_summary_integer_instruction,
            "lettergrade" : command_summary_lettergrade_instruction
        },
        "qa" : {
            "integer" : command_qa_integer_instruction,
            "lettergrade" : command_qa_lettergrade_instruction
        }
    },
    "llama2" : {
        "summary" : {
            "integer" : llama2_summary_integer_instruction,
            "lettergrade" : llama2_summary_lettergrade_instruction
        },
        "qa" : {
            "integer" : llama2_qa_integer_instruction,
            "lettergrade" : llama2_qa_lettergrade_instruction
        }
    },
}


# def calibrate_evaluator(
#     evaluator_name, 
#     task,
#     feedback,
#     calibrator_model=gpt4, 
#     baseline_instruction=None
# ):

#     instruction = baseline_instruction
#     if baseline_instruction is None:
#         instruction = baseline_instructions[task][feedback]

#     calibrator = LLMChain(llm=calibrator_model, prompt=prompt_calibration_prompt)

#     while True:
#         print("current instruction:", instruction)

#         print("---------------------------------------------")
#         calibrator_input = {"current_instruction" : instruction, "input_text" : sample_inputs[task]}
#         for candidate in ["perfect", "ok", "awful", "irrelevant"]:
#             evaluator_input = {
#                 "instruction" : instruction,
#                 "input_text" : sample_inputs[task],
#                 "output_text" : sample_outputs[task][candidate]
#             }
#             if task == "qa":
#                 evaluator_input["context"] = sample_article
#             evaluation = evaluators[task][feedback][evaluator_name](evaluator_input)["text"]
#             parsed_evaluation = grade_parser[feedback](evaluation)
#             print("-\nunparsed\n-\n", evaluation)
#             print("-\nparsed\n-\n", parsed_evaluation)

#             calibrator_input[f"candidate_{candidate}"] = sample_outputs[task][candidate]
#             calibrator_input[f"expected_{candidate}"] = expected_calibrated_results[feedback][candidate]
#             calibrator_input[f"evaluation_{candidate}"] = parsed_evaluation

#         quit = input("-----\nPress q if this is calibrated enough. Otherwise just hit enter:\n>>>")

#         if quit == "":
#             calibrated_instruction = calibrator(calibrator_input)["text"]   
#             print(f"----\n{calibrated_instruction}")
#             user_edited_calibrated_instruction = input("-----\nIf the new instruction needs to be parsed, type it now. Otherwise just hit enter:\n>>>")
#             if user_edited_calibrated_instruction == "":
#                 instruction = calibrated_instruction
#             else:
#                 instruction = user_edited_calibrated_instruction
#         else:
#             break

