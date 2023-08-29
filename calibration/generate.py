import re
import time
import pandas as pd
import json
import numpy as np
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback

from shared_prompts import tasks, gpt35, GPT35_LETTER_FEW_SHOT, GPT35_BASE_ZERO_SHOT
# TODO: standardize w Max's code

class SummaryEvaluator:
    def __init__(self,prompt, llm, regex):
        self.prompt = prompt
        self.regex = regex
        self.chain = LLMChain(llm=llm, prompt=prompt)
        self.generation_times = []
        self.tokens = []

    def parse(self, response):
        match = re.search(self.regex, response.upper())
        if match is None:
            print("LLM evaluator returned invalid response")
            return None
        return match.group()
    
    def write_run_report(self, name, total_tokens, total_cost):
        with open(f'{name}_stats.json', 'w') as f:
            json.dump({'max_latency': np.max(self.generation_times), 
                       'min_latency': np.min(self.generation_times),
                       'avg_latency': np.mean(self.generation_times),
                       'total_tokens': total_tokens,
                       'total_cost': total_cost}, f)
        self.generation_times = []
        self.tokens = []
    
    def run(self, articles, summaries, name):
        scores = []
        with get_openai_callback() as cb:
            for article, summary in zip(articles, summaries):
                inputs = {
                    'input_text': article,
                    'output_text': summary
                }
                start = time.time()
                resp = self.chain(inputs)
                end = time.time()
                scores.append(self.parse(resp["text"]))
                self.generation_times.append(end - start)
            self.write_run_report(name, cb.total_tokens, cb.total_cost)
        return scores
    

base_evaluator = SummaryEvaluator(prompt=GPT35_BASE_ZERO_SHOT, llm=gpt35, regex=r'(YES|NO)')
letter_evaluator = SummaryEvaluator(prompt = GPT35_LETTER_FEW_SHOT, llm=gpt35, regex=r'\b([A-E][+-]?|F)(?!\w)')


# for the "train" samples do multiple generations to measure variance
data = pd.read_csv('./data/xsum_humaneval/summaries_train.csv')
articles = data['input_article'].tolist()
summaries = data['summary'].tolist()

for i in range(1):
    # print("running base evaluator")
    base_scores = base_evaluator.run(articles, summaries, f'gpt35_score_{i}')
    data[f'gpt35_score_{i}'] = base_scores
    data.to_csv('evaluated_summaries_train_base.csv')
    print("running letter grade evaluator")
    letter_scores = letter_evaluator.run(articles, summaries, f'gpt35_letter_{i}')
    data[f'gpt35_letter_{i}'] = letter_scores
    data.to_csv('evaluated_summaries_train_letter_4shot.csv')
# print("running calibrated evaluator")
# engineered_scores = engineered_evaluator.run(articles, summaries)
# data['gpt35_letter_opt'] = engineered_scores
# data.to_csv('evaluated_summaries_letter_opt.csv')


# generate a single response for the remainder of the data
data = pd.read_csv('./data/xsum_humaneval/summaries_test.csv')
articles = data['input_article'].tolist()
summaries = data['summary'].tolist()
print("running base evaluator on test")
base_scores = base_evaluator.run(articles, summaries, 'gpt35_score_test')
data[f'gpt35_score'] = base_scores
data.to_csv('evaluated_summaries_test_base.csv')
print("running letter grade evaluator on test")
letter_scores = letter_evaluator.run(articles, summaries, 'gpt35_letter_test')
data[f'gpt35_letter'] = letter_scores
data.to_csv('evaluated_summaries_test_letter_4shot.csv')

