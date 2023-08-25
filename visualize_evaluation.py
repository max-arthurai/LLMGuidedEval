import matplotlib.pyplot as plt
import os
import pandas as pd
from time import time

tasks = ["summary", "qa"]
models = ["gpt35", "gpt4", "claude", "command", "llama2"]
evaluators = ["gpt35"]

n_data = 5
n_attempts = 3

candidate_output_columns = {
    "summary" : ["ground_truth_summary"]+[f"{m}_{g}_{i}" for i in range(n_attempts) for g in ["summary", "rephrase_gt"] for m in models],
    "qa" : ["ground_truth_answer"]+[f"{m}_{g}_{i}" for i in range(n_attempts) for g in ["answer", "rephrase_gt"] for m in models],
}

evaluation_directories = {"summary" : "summarization/old_summary_evaluations", "qa" : "rag_qa/old_answer_evaluations"}

results = {}

for evaluator in evaluators:
    results[evaluator] = {}
    for task in tasks:
        results[evaluator][task] = {}

        # direct feedback
        for feedback in ["written", "binary", "integer", "lettergrade"]:
            results[evaluator][task][feedback] = {}
            for candidate in candidate_output_columns[task]:
                results[evaluator][task][feedback][candidate] = []
                for data_num in range(n_data):
                    filename = f"data/{evaluation_directories[task]}/{evaluator}-{feedback}-{candidate}-{data_num}.txt"
                    if os.path.exists(filename):
                        with open(filename) as f:
                            results[evaluator][task][feedback][candidate].append(f.read())
                        
        # A/B testing feedback 
        results[evaluator][task]["abtest"] = {}
        for candidate_A in candidate_output_columns[task]:
            results[evaluator][task]["abtest"][candidate_A] = {}
            other_models = [m for m in candidate_output_columns[task] if m != candidate_A]
            for candidate_B in other_models:
                results[evaluator][task]["abtest"][candidate_A][candidate_B] = []
                for data_num in range(n_data):
                    filename = f"data/{evaluation_directories[task]}/{evaluator}-abtest-{candidate_A}-{candidate_B}-{data_num}.txt"
                    # only consider candidates from attempt # 0 to narrow the space for now
                    if os.path.exists(filename) and "_1" not in candidate_A and "_1" not in candidate_B and "_2" not in candidate_A and "_2" not in candidate_B:
                        with open(filename) as f:
                            results[evaluator][task]["abtest"][candidate_A][candidate_B].append(f.read())

# TODO
