import tqdm
import pandas as pd
from load_fever_dataset import load_fever_dataset_exclude_NEI
from utils.load_data import find_text_by_ids, random_get_texts_exclude_ids

# Load the dataset once
dataset = load_fever_dataset_exclude_NEI('train.jsonl')

data_length = len(dataset[:100000])
reranker_trainset_data = []

neg_texts_amount = 7

for i in tqdm.tqdm(range(data_length), desc='generate trainset'):
    evidence_row = dataset.iloc[i]
    query = evidence_row['claim']
    true_evidences = evidence_row['evidence']

    pos = []
    exclude_ids = set()
    
    for evidence in true_evidences:
        true_evidence_idx = evidence[0][2]
        true_evidence_sentence = find_text_by_ids(true_evidence_idx)
        
        if true_evidence_sentence:
            pos.append(true_evidence_sentence)
            exclude_ids.add(true_evidence_idx)

    if pos:
        neg = random_get_texts_exclude_ids(neg_texts_amount, exclude_ids)
        reranker_trainset_data.append({'query': query, 'pos': pos, 'neg': neg})

# Create the DataFrame once
reranker_trainset = pd.DataFrame(reranker_trainset_data)

# Save reranker_trainset to jsonl
reranker_trainset.to_json('reranker_trainset.jsonl', orient='records', lines=True, force_ascii=False)
