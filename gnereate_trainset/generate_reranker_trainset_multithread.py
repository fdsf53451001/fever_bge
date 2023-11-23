import pandas as pd
from load_fever_dataset import load_fever_dataset_exclude_NEI
from utils.load_data import find_text_by_ids, random_get_texts_exclude_ids
import concurrent.futures
import tqdm
import sys


def process_data(i):
    evidence_row = dataset.iloc[i]
    query = evidence_row['claim']
    true_evidences = list(evidence_row['evidence'])[0]
    
    evidence_amount = len(true_evidences)
    pos = []
    exclude_ids = []
    for k in range(evidence_amount):
        true_evidence_idx = true_evidences[k][2]
        true_evidence_sentence = find_text_by_ids(true_evidence_idx)
        if true_evidence_sentence:
            pos.append(true_evidence_sentence)
            exclude_ids.append(true_evidence_idx)
    
    if not pos:
        return None

    neg = random_get_texts_exclude_ids(neg_texts_amount, exclude_ids)

    return {'query': query, 'pos': pos, 'neg': neg}

if __name__ == "__main__":
    dataset = load_fever_dataset_exclude_NEI('fever/trainset/train.jsonl')
    data_length = len(dataset[:100000])

    reranker_trainset = pd.DataFrame(columns=['query', 'pos', 'neg'])
    neg_texts_amount = 7

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_data, data) for data in range(data_length)]

        with tqdm.tqdm(total=data_length, desc='generate trainset') as pbar:
            for future in futures:
                # Wait for the task to complete
                future.result()
                pbar.update(1)  # Update progress bar
                sys.stdout.flush()  # Add this line

    # Filter out None values
    results = [future.result() for future in futures if future.result() is not None]

    for result in results:
        reranker_trainset = pd.concat([reranker_trainset, pd.DataFrame([result])], ignore_index=True)

    # Save reranker_trainset to jsonl
    reranker_trainset.to_json('fever/trainset/reranker_trainset.jsonl', orient='records', lines=True, force_ascii=False)
