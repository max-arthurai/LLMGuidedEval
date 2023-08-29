import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LETTER_TO_INT = {
    'F': 0,
    'D-': 1,
    'D': 2,
    'D+': 3,
    'C-': 4, 
    'C': 5,
    'C+': 6,
    'B-': 7,
    'B': 8,
    'B+': 9,
    'A-': 10,
    'A': 11,
    'A+': 12
}

BINARY_TO_INT = {
    'NO': 0,
    'YES': 1
}

def calculate_ranking_metric(score_min, score_max, human_col, score_col, k, filepath, score_map):
    df = pd.read_csv(filepath)
    relevant_documents = df[(score_min <= df[human_col]) & (df[human_col] < score_max)]['document_id'].tolist()
    print(len(relevant_documents))
    retrieved_documents = df[df[human_col] >= score_min].sort_values(by=score_col, key=lambda x: x.map(score_map))[:k]['document_id'].tolist()
    count = 0
    for doc in retrieved_documents:
        if doc in relevant_documents:
            count += 1
    score = count / k
    return score

all_model_scores = {}
evaluators = ['gpt35_score', 'gpt35_letter', 'gpt35_letter_opt']
# thresholds = [(0, 0.4,), (0.4, 0.7)]
thresholds = [(1.0, 1.4), (1.4, 1.7), (1.7, 2.0), (2.0, 2.4), (2.4, 2.7), (2.7, 3.0), (3.0, 3.4), (3.4, 3.7), (3.7, 4.0), (4.0, 4.4), (4.4, 4.7)]
for evaluator in evaluators:
    ranking_scores = []
    for threshold in thresholds:
        score_map = BINARY_TO_INT if evaluator == 'gpt35_score' else LETTER_TO_INT
        ranking_scores.append(calculate_ranking_metric(score_min=threshold[0], score_max=threshold[1], human_col='relevance', score_col=evaluator, k=5, filepath='./evaluated_summaries_letter_opt.csv', score_map=score_map))
    all_model_scores[evaluator] = ranking_scores


x = np.arange(len(thresholds))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for attribute, measurement in all_model_scores.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Precision at 5')
ax.set_title('Precision at k by model evaluator (relevance)')
ax.set_xticks(x + width, thresholds )
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 1)

# plt.show()
fig.savefig('relevance_at_5.png')