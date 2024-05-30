import json
import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

avg_seq_len = 0
num_seq = 0
with open('dataset/train.jsonl', 'r') as f:
    lines = f.readlines()
    for l in tqdm.tqdm(lines):
        data = json.loads(l)
        claim = 'Represent this sentence for searching relevant passages:'
        claim += data['claim']
        input_ids = tokenizer(claim)['input_ids']

        avg_seq_len += len(input_ids)
        num_seq += 1

print('average seq len:', avg_seq_len/num_seq)