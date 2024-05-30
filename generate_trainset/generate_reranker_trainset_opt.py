import tqdm
import pandas as pd
import sys
sys.path.append("/home/jovyan/fever_bge")
from load_fever_dataset import load_fever_dataset_exclude_NEI
from utils.load_data import find_text_by_ids
# from utils.load_data import random_get_texts_exclude_ids

# load retrieval dataset
retrieval_pd = pd.read_json('doc_eval/result/trainset_evidence_100.jsonl', lines=True)
reranker_pd = pd.read_json('doc_eval/result/trainset_evidence_rerank_10.jsonl', lines=True)

def get_neg_example_from_retrieval(id:str, exclude_ids:list, amount=5) -> list:
    evidences : list = retrieval_pd[retrieval_pd['id']==id]['evidences'].values[0]
    evidences = [e for e in evidences if e not in exclude_ids]
    return evidences[:amount]

def get_example_from_reranker(id:str, amount=5) -> list:
    evidences : list = reranker_pd[reranker_pd['id']==id]['evidences'].values[0]
    return evidences[:amount]

# get_neg_example_from_retrieval(75397, ['21st_Century_Fox'])

# Load the dataset once
dataset = load_fever_dataset_exclude_NEI('dataset/train.jsonl')

data_length = len(dataset)
reranker_trainset_data = []

neg_texts_amount = 30

for i in tqdm.tqdm(range(data_length), desc='generate trainset'):
    evidence_row = dataset.iloc[i]
    id = evidence_row['id']
    query = evidence_row['claim']
    true_evidences = evidence_row['evidence']

    pos = []
    neg = []
    exclude_ids = set()
    
    for evidence in true_evidences:
        true_evidence_idx = evidence[0][2]
        true_evidence_sentence = find_text_by_ids(true_evidence_idx)
        
        if true_evidence_sentence:
            pos.append(true_evidence_sentence)
            exclude_ids.add(true_evidence_idx)

    if pos:
        # neg = get_neg_example_from_retrieval(id, exclude_ids, amount=neg_texts_amount)
        # neg = [(lambda x:x if x else "")(find_text_by_ids(e)) for e in neg]
        # reranker_trainset_data.append({'query': query, 'pos': pos, 'neg': neg})

        
        neg_ids = get_neg_example_from_retrieval(id, exclude_ids, amount=neg_texts_amount)
        reranker_pos_ids = get_example_from_reranker(id, amount=4)
        # pos_plus_ids = neg_ids[:3]
        for p in reranker_pos_ids:
            if p not in exclude_ids:
                p_texts = find_text_by_ids(p)
                if p_texts:
                    pos.append(p_texts)
                    exclude_ids.add(p)
        # neg_ids = neg_ids[3:]
        for n in neg_ids:
            if n not in exclude_ids:
                n_texts = find_text_by_ids(n)
                if n_texts:
                    neg.append(n_texts)
        # neg = [(lambda x:x if x else "")(find_text_by_ids(e)) for e in neg]
        reranker_trainset_data.append({'query': query, 'pos': pos, 'neg': neg})

# Create the DataFrame once
reranker_trainset = pd.DataFrame(reranker_trainset_data)

# Save reranker_trainset to jsonl
reranker_trainset.to_json('genreate_trainset/reranker_trainset.jsonl', orient='records', lines=True, force_ascii=False)
