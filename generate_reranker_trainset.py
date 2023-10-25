import tqdm
import pandas as pd
from load_fever_dataset import load_fever_dataset_exclude_NEI
from utils.load_data import find_text_by_ids, random_get_texts_exclude_ids


dataset = load_fever_dataset_exclude_NEI('train.jsonl')
data_length = len(dataset[:100000])

reranker_trainset = pd.DataFrame(columns=['query','pos','neg'])
neg_texts_amount = 7

for i in tqdm.tqdm(range(data_length),desc='generate trainset'):
    evidence_row = dataset.iloc[i]
    query = evidence_row['claim']
    true_evidences = list(dataset['evidence'])[0]
    
    evidence_amount = len(true_evidences)
    pos = []
    exclude_ids = []
    for k in range(evidence_amount):
        true_evidence_idx = true_evidences[k][0][2]
        true_evidence_sentence = find_text_by_ids(true_evidence_idx)
        if true_evidence_sentence:
            pos.append(true_evidence_sentence)
            exclude_ids.append(true_evidence_idx)
    
    if not pos:
        continue

    neg = random_get_texts_exclude_ids(neg_texts_amount, exclude_ids)

    # save to reranker_trainset
    row = {'query':query,'pos':pos,'neg':neg}
    reranker_trainset = pd.concat([reranker_trainset, pd.DataFrame([row])], ignore_index=True)

# save reranker_trainset to jsonl
reranker_trainset.to_json('reranker_trainset.jsonl',orient='records',lines=True, force_ascii=False)


    
