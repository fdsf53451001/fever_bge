import tqdm
import json

# read jsonl file
with open('verify/dataset/created/testset_evidences_10doc.jsonl', 'r') as f:
    data = f.readlines()

evidence_missing = 0
with open('verify/dataset/created/testset_evidences_spilit_10doc.jsonl', 'w') as f:
    for row in tqdm.tqdm(data[:]):
        row = json.loads(row)

        id = row['id']
        claim = row['claim']
        evidences = row['evidences']

        for evidence in evidences:     
            row_processed = {'id': id, 'claim': claim, 'evidence': evidence}        
            f.write(json.dumps(row_processed) + '\n')