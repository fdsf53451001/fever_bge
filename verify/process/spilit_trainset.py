import tqdm
import json

# read jsonl file
with open('verify/dataset/created/train_processed.jsonl', 'r') as f:
    data = f.readlines()

evidence_missing = 0
with open('verify/dataset/created/train_processed_spilit.jsonl', 'w') as f:
    for row in tqdm.tqdm(data[:]):
        row = json.loads(row)

        label = row['label']
        claim = row['claim']
        evidences = row['evidence']

        for evidence in evidences:     
            row_processed = {'label': label, 'claim': claim, 'evidence': evidence}        
            f.write(json.dumps(row_processed) + '\n')