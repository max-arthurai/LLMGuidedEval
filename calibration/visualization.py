import pandas as pd

class CalibrationPlotter:
    def __init__(self, bin_column, bins=None):
        self.bins = bins
        self.bin_column = bin_column

    def plot(self, filepath):
        df = pd.read_csv(filepath)
        bin_groups = df.groupby(self.bin_column)

        faithfulness = bin_groups['faithfulness'].mean()
        fig = faithfulness.plot(kind='bar', title=f'Faithfulness {self.bin_column}', ylabel='Avg Human Faithfulness Score', xlabel='Model Evaluator Score').get_figure()
        fig.savefig(f'Faithfulness {self.bin_column}.png')

        relevance = bin_groups['relevance'].mean()
        fig = relevance.plot(kind='bar', title=f'Relevance {self.bin_column}', ylabel='Avg Human Relevance Score', xlabel='Model Evaluator Score').get_figure()
        fig.savefig(f'Relevance {self.bin_column}.png')

        coherence = bin_groups['coherence'].mean()
        fig = coherence.plot(kind='bar', title=f'Coherence {self.bin_column}', ylabel='Avg Human Coherence Score', xlabel='Model Evaluator Score').get_figure()
        fig.savefig(f'Coherence {self.bin_column}.png')



# CalibrationPlotter(bin_column='gpt35_score').plot('./evaluated_summaries_letter_opt.csv')
# CalibrationPlotter(bin_column='gpt35_letter').plot('./evaluated_summaries_letter_opt.csv')
CalibrationPlotter(bin_column='gpt35_letter_opt').plot('./evaluated_summaries_letter_opt.csv')