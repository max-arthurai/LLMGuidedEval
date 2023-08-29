import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
Aggregate all summary rating across models and shots. Average the 3 ratings per summary
Requires downloading helm xsum humaneval dataset
"""

data = {
    'input_article': [],
    'summary': [],
    'model_name': [],
    'faithfulness': [],
    'relevance': [],
    'coherence': []
}

for i in range(5):
    df = pd.read_csv(f'./annotations/xsum_{i}shots.csv.tmp')
    for group in df.groupby('HITId'):
        data['input_article'].append(group[1]["Input.source_article"].iloc[0])
        data['summary'].append(group[1]["Input.output_text"].iloc[0])
        data['model_name'].append(group[1]['Input.model_name'].iloc[0])
        data['faithfulness'].append(group[1]["Answer.consistency.consistent"].mean())

        scores = []
        for _, row in group[1].iterrows():
            for i in range(1, 6):
                if row[f"Answer.relevance.rel_{i}"]:
                    scores.append(i)
                    break
        data['relevance'].append(np.mean(scores))

        scores = []
        for _, row in group[1].iterrows():
            for i in range(1, 6):
                if row[f"Answer.coherence.cohere_{i}"]:
                    scores.append(i)
                    break
        data['coherence'].append(np.mean(scores))

processed_df = pd.DataFrame(data)
processed_df.to_csv('summaries_processed.csv')

def split(df):
    train, test = train_test_split(df, test_size=500, random_state=42)
    train.to_csv('summaries_train.csv')
    test.to_csv('summaries_test.csv')


# split the dataset, we will use the smaller split to compute multiple generations
split(processed_df)

# sampled_df = processed_df.sample(n=300, random_state=42)

# print(sampled_df['relevance'].value_counts())
# print(sampled_df['coherence'].value_counts())
# print(sampled_df['faithfulness'].value_counts())

# sampled_df.to_csv('summaries_sampled.csv')