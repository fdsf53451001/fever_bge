import json
import tqdm
import pandas as pd

from load_fever_dataset import load_fever_dataset, load_fever_dataset_exclude_NEI

top_evidenct_amount = 10
evidence_df = load_fever_dataset('doc_eval/result/devset_evidence_rerank_10.jsonl')
gold_devset_df = load_fever_dataset_exclude_NEI('dataset/shared_task_dev.jsonl')

data_length = len(evidence_df)
match_amount = 0

compare_wiki_format = False

for i in tqdm.tqdm(range(data_length),desc='scoring'):
    evidence_row = evidence_df.iloc[i]
    id = evidence_row['id']
    gold_devset_row = gold_devset_df[gold_devset_df['id']==id]
    # remove NEI
    if not len(gold_devset_row):
        data_length -= 1
        continue

    evidences_idx = list(evidence_row['evidences'])
    evidences_idx = evidences_idx[:top_evidenct_amount]
    if compare_wiki_format:
        # the respond title of wiki api likes "Murda Beatz" is not the same as the title in wiki dump, replace space with _ to match the title in wiki dump
        # turn this on with the wiki api
        evidences_idx = [evidence.replace(' ','_').replace('(','-LRB-').replace(')','-RRB-') for evidence in evidences_idx]

    gold_evidence_sets = list(gold_devset_row['evidence'])[0]  
    for evidence_set in gold_evidence_sets:
        gold_evidences_idxs = [evidence[2] for evidence in evidence_set]

        for gold_evidence_idx in gold_evidences_idxs:
            if not gold_evidence_idx in evidences_idx:
                break
        else:
            match_amount += 1
            break

print('evidence amount', top_evidenct_amount)
print('data_length', data_length)
print('match', match_amount)
print('recall', match_amount/data_length)