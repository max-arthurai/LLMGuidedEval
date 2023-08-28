import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import pandas as pd
import re
from time import time

# define evaluation parsers
def get_number_grade(s: str) -> float:
    # This regex pattern will match numbers 0-10 with optional decimal values for numbers other than 10.
    pattern = r'\b(10|[0-9](?:\.\d)?)\b'
    match = re.search(pattern, s)
    return float(match.group()) if match else -1

def get_letter_grade(s: str) -> str:
    # This regex pattern will match letter grades A+ to F, with optional + or -.
    pattern = r'\b([A-E][+-]?|F)(?!\w)'
    match = re.search(pattern, s)
    return match.group() if match else "nan"
grade_parser = {"integer" : get_number_grade, "lettergrade" : get_letter_grade, "written" : lambda x : x}

tasks = ["summary", "qa"]
models = ["gpt35", "gpt4", "claude", "command", "llama2"]
evaluators = ["gpt35", "claude", "command", "llama2"]

n_data = 5
n_attempts = 3

candidate_output_columns = {
    "summary" : sorted([f"{m}_summary_{i}" for i in range(n_attempts) for m in models])+sorted([f"{m}_rephrase_gt_{i}" for i in range(n_attempts) for m in models]) + ["ground_truth_summary"],
    "qa" : sorted([f"{m}_answer_{i}" for i in range(n_attempts) for m in models]) + sorted([f"{m}_rephrase_gt_{i}" for i in range(n_attempts) for m in models]) + ["ground_truth_answer"],
}

evaluation_directories = {
    "summary" : "summarization/summary_evaluations", 
    "qa" : "rag_qa/answer_evaluations"
}

results = {}

for evaluator in evaluators:
    results[evaluator] = {}
    for task in tasks:
        results[evaluator][task] = {}
        for feedback in ["integer", "lettergrade"]:
            results[evaluator][task][feedback] = {}
            for candidate in candidate_output_columns[task]:
                results[evaluator][task][feedback][candidate] = []
                for data_num in range(n_data):
                    filename = f"data/{evaluation_directories[task]}/{evaluator}-{feedback}-{candidate}-{data_num}.txt"
                    if os.path.exists(filename):
                        with open(filename) as f:
                            results[evaluator][task][feedback][candidate].append(grade_parser[feedback](f.read()))
                        
        # # A/B testing feedback 
        # results[evaluator][task]["abtest"] = {}
        # for candidate_A in candidate_output_columns[task]:
        #     results[evaluator][task]["abtest"][candidate_A] = {}
        #     other_models = [m for m in candidate_output_columns[task] if m != candidate_A]
        #     for candidate_B in other_models:
        #         results[evaluator][task]["abtest"][candidate_A][candidate_B] = []
        #         for data_num in range(n_data):
        #             filename = f"data/{evaluation_directories[task]}/{evaluator}-abtest-{candidate_A}-{candidate_B}-{data_num}.txt"
        #             # only consider candidates from attempt # 0 to narrow the space for now
        #             if os.path.exists(filename) and "_1" not in candidate_A and "_1" not in candidate_B and "_2" not in candidate_A and "_2" not in candidate_B:
        #                 with open(filename) as f:
        #                     results[evaluator][task]["abtest"][candidate_A][candidate_B].append(f.read())

def get_target_values(df, target):
    values = []
    for c in df.columns:
        if target in c:
            values.extend(df[c].values.tolist())
    return values

all_grades = ["A+", "A", "A-", "B+", "B", "B-", "C+", "C", "C-", "D+", "D", "D-", "F"]

def get_target_value_counts(df, target):
    value_counts = {g : 0 for g in all_grades}
    for c in df.columns:
        if target in c:
            values = df[c].values.tolist()
            for v in values:
                value_counts[v] += 1
    return value_counts

def plot_evaluation(
    evaluator,
    task,
    feedback
):
    grid = pd.DataFrame(results[evaluator][task][feedback])[candidate_output_columns[task]]

    # visualize all scores
    fig, ax = plt.subplots(figsize=(10,8))
    grid_colors = grid
    if feedback == "integer":
        im = ax.imshow(grid, cmap='coolwarm_r')
        im.set_clim([0, 10])
    elif feedback == "lettergrade":
        grid_colors = grid.replace({
            "A+" : 10, "A" : 9, "A-" : 8, "B+" : 7, "B" : 6, "B-" : 5, "C+" : 4, "C" : 3, "C-" : 2, "D+" : 1, "D" : 1, "D-" : 1, "F" : 0
        })
        im = ax.imshow(grid_colors, cmap='coolwarm_r')
        im.set_clim([1, 10]) # no Fs observed yet

    # Annotation loop
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.iloc[i, j]
            ax.text(j, i, str(value).replace(".0",""), ha='center', va='center', color='w' if (feedback == "integer" and value <= 5) or (feedback == "lettergrade" and grid_colors.iloc[i, j] <= 5) else 'k')
    
    # 5 models x (3 direct, 3 rephrase) + 1 ground truth = 31 candidates
    plt.xticks(np.arange(len(models) * n_attempts * 2 + 1), [x.replace("_summary", "").replace("_answer", "").replace("_rephrase", "") for x in candidate_output_columns[task]], rotation=90)
    ax.set_xlabel(f"Candidate output for {task}")
    ax.set_yticks(np.arange(n_data))
    ax.set_ylabel(f"{task} test cases")
    ax.set_title(f"{evaluator} {task} {feedback} evaluation")
    plt.tight_layout()
    plt.savefig(f"plots/{evaluator}/{evaluator}_{task}_{feedback}.png")
    plt.close()

    # summary bar plot of score count for each target
    if feedback == "integer":
        fig, ax = plt.subplots(figsize=(12,6))
        for model_num, m in enumerate(sorted(models)):
            scores = [x for x in get_target_values(grid, m) if x>=0]
            ax.bar([model_num], [sum(scores)/len(scores)], yerr=np.array(scores).std())
        scores = [x for x in get_target_values(grid, "ground_truth") if x>=0]
        heights = [sum(scores)/len(scores)]
        ax.bar([len(models)], heights, yerr=np.array(scores).std())
        ax.set_ylim([0.8*min(heights),1.2*max(heights)])
        plt.xticks(np.arange(len(models)+1), sorted(models) + ["ground_truth"], rotation=90)
        ax.set_xlabel('Candidate output')
        ax.set_ylabel(f'{evaluator} {feedback} evaluation')
        ax.set_title(f'{evaluator} {task} {feedback} evaluation distribution')
        plt.tight_layout()
        plt.savefig(f"plots/{evaluator}/bar_{evaluator}_{task}_{feedback}.png")

    elif feedback == "lettergrade":
        fig, axes = plt.subplots(1, len(models)+1, figsize=(30,4), sharex=True, sharey=True)
        fig.suptitle(f"{evaluator} {task} letter grade evaluation")
        for model_num, m in enumerate(sorted(models)):
            grades = get_target_value_counts(grid, m)
            total_count = sum(grades.values())
            bars = axes[model_num].bar(np.arange(len(grades)), list(grades.values()))
            axes[model_num].set_xticks(np.arange(len(grades)))
            axes[model_num].set_xticklabels(list(grades.keys()))
            axes[model_num].set_yticks([0,5,10,15,20,25])
            axes[model_num].set_title(f"grade distribution for {m}")

            # Annotating bars with percentage
            for bar, value in zip(bars, grades.values()):
                percentage = (value / total_count) * 100
                if percentage > 0:
                    axes[model_num].text(bar.get_x() + bar.get_width() / 2,
                                        bar.get_height() + 0.5, 
                                        f'{percentage:.1f}%', 
                                        ha='center', 
                                        va='bottom')
    
        gt_grades = get_target_value_counts(grid, "ground_truth")
        gt_total_count = sum(gt_grades.values())
        gt_bars = axes[len(models)].bar(np.arange(len(gt_grades)), list(gt_grades.values()))
        axes[len(models)].set_xticks(np.arange(len(gt_grades)))
        axes[len(models)].set_xticklabels(list(gt_grades.keys()))
        axes[len(models)].set_yticks([0,5,10,15,20,25])
        axes[len(models)].set_title(f"{evaluator} grade distribution for ground truth")

        # Annotating bars with percentage
        for bar, value in zip(gt_bars, gt_grades.values()):
            percentage = (value / gt_total_count) * 100
            if percentage > 0:
                axes[len(models)].text(bar.get_x() + bar.get_width() / 2,
                                    bar.get_height() + 0.5, 
                                    f'{percentage:.1f}%', 
                                    ha='center', 
                                    va='bottom')
        plt.tight_layout()
        plt.savefig(f"plots/{evaluator}/bar_{evaluator}_{task}_{feedback}.png")

plot_evaluation("gpt35", "summary", "integer")
plot_evaluation("gpt35", "summary", "lettergrade")
plot_evaluation("gpt35", "qa", "integer")
plot_evaluation("gpt35", "qa", "lettergrade")
plot_evaluation("claude", "summary", "integer")
plot_evaluation("claude", "summary", "lettergrade")
plot_evaluation("claude", "qa", "integer")
plot_evaluation("claude", "qa", "lettergrade")
plot_evaluation("command", "summary", "integer")
plot_evaluation("command", "summary", "lettergrade")
plot_evaluation("command", "qa", "integer")
plot_evaluation("command", "qa", "lettergrade")
plot_evaluation("llama2", "summary", "integer")
plot_evaluation("llama2", "summary", "lettergrade")
plot_evaluation("llama2", "qa", "integer")
plot_evaluation("llama2", "qa", "lettergrade")