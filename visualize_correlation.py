import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

human_feedback_to_plot = ['faithfulness', 'relevance', 'coherence']

def plot_human_scores_against_binary_llm_feedback(name, df, binary_feedback_col):
	plt.figure(figsize=(15, 5))
	plt.suptitle(f'Human feedback vs binary score from gpt-3.5-turbo', fontsize=20)
	
	for i, col in enumerate(human_feedback_to_plot, 1):
	    plt.subplot(1, 3, i)
	    sns.violinplot(x=df[binary_feedback_col], y=df[col])
	    plt.title(f'Human {col} scores \nvs\n "Is this {col.replace("nce", "nt").replace("ness", "")}?" prompted feedback')
	    plt.ylim([df[col].min(), df[col].max()])

	plt.tight_layout()
	plt.savefig(f"{name}.png")

def plot_human_scores_against_lettergrade_llm_feedback(name, df, lettergrade_feedback_col):
	plt.figure(figsize=(10, 10))
	plt.suptitle(f'Human feedback vs letter grade from gpt-3.5-turbo', fontsize=20)
	for i, col in enumerate(human_feedback_to_plot, 1):
		plt.subplot(3, 1, i)

		# Map strings to numbers for jittering
		unique_vals = df[lettergrade_feedback_col].unique()
		mapping = {val: i for i, val in enumerate(unique_vals)}
		x_numeric = df[lettergrade_feedback_col].map(mapping)

		# Add jitter to the data
		x_jitter = x_numeric + (np.random.rand(len(df)) * 0.2 - 0.1)  # Random values between -0.05 and 0.05
		y_jitter = df[col] + (np.random.rand(len(df)) * 0.2 - 0.1)

		sns.scatterplot(x=x_jitter, y=y_jitter, alpha=0.3)

		# Set the x-ticks to original string values
		plt.xticks(ticks=range(len(unique_vals)), labels=unique_vals)

		plt.title(f'Human {col} scores \nvs\n "Is this {col.replace("nce", "nt").replace("ness", "")}?" prompted feedback')
		plt.ylim([df[col].min(), df[col].max()])

	plt.tight_layout()
	plt.savefig(f"{name}.png")

base_df = pd.read_csv("summary_evaluations/evaluated_summaries_test_base.csv")
plot_human_scores_against_binary_llm_feedback("base", base_df, "gpt35_score")
letter_df = pd.read_csv("summary_evaluations/evaluated_summaries_test_letter.csv").sort_values(by='gpt35_letter')
plot_human_scores_against_lettergrade_llm_feedback("letter", letter_df, "gpt35_letter")


