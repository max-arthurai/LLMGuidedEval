import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

"""
Requires downloaded llama weights
"""

tokenizer = LlamaTokenizer.from_pretrained("./LLMGuidedEval/calibration/llama_hf")
model = LlamaForCausalLM.from_pretrained("./LLMGuidedEval/calibration/llama_hf", torch_dtype=torch.float16).to('cuda')

YES_TOKEN_IDS = {3869, 4874}
NO_TOKEN_IDS = {694, 1939}
def generate_with_llama(text):
    # generate
    inputs = tokenizer(text, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_new_tokens=5, return_dict_in_generate=True, output_scores=True)
    
    # get token probabilities and remove prompt tokens
    scores = model.compute_transition_scores(outputs.sequences, outputs.scores, normalize_logits=True).squeeze()
    new_tokens = outputs.sequences[:, inputs['input_ids'].shape[1]:]
    print(new_tokens)
    
    # merge tokens
    p_yes = 0
    yes_found = False
    for i, token in enumerate(new_tokens.squeeze()):
        if token.item() in YES_TOKEN_IDS:
            p_yes = scores[i]
            yes_found = True
        elif token.item() in NO_TOKEN_IDS:
            p_yes = 1 - scores[i]
            yes_found = True
    if not yes_found:
        print("LLM evaluator return invalid response")
    
    return p_yes, tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

LLAMA_SYSTEM_PROMPT = """
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. You are evaluating the quality of a summary. 
<</SYS>>
\n=Instruction=\n{instruction}
\n=Article=\n{input_text}\n=Summary=\n{output_text}
\n=Answer=\n [/INST]
"""

prompt = LLAMA_SYSTEM_PROMPT.format(instruction="Is this summary faithful, coherent, and relevant? Only respond with 'yes' or 'no', nothing else.", input_text="yesterday was sunny", output_text="yesterday was sunny")

prob, text = generate_with_llama(prompt.strip())


# import runhouse as rh
# cluster = rh.cluster(
#               name="rowan_runhose",
#               host=['150.136.49.89'],
#               ssh_creds={'ssh_user': 'ubuntu', 'ssh_private_key': '~/.ssh/id_rsa'}
# )
# # cluster.up()

# def getpid(a=0, b=0):
#     import os
#     return "test"

# cluster.ssh()

# getpid_remote = rh.function(fn=getpid).to(system=cluster)

# print(f"local function result: {getpid()}")
# print(f"remote function result: {getpid_remote(stream_logs=True)}")