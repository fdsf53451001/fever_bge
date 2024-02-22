import json
import tqdm

def split_lines(lines):
    lines = lines.split('\n')  # 以'\n'分割字符串成行
    result = []

    for line_number, line in enumerate(lines):
        if line:
            parts = line.split('\t', 1)[1]  # 以'\t'分割行成部分
            # result.append((line_number, parts))
            result.append(parts)

    return result

# read jsonl file
with open('verify/dataset/train_golden.jsonl', 'r') as f:
    data = f.readlines()

evidence_missing = 0
with open('verify/dataset/created/train_processed_NEI.jsonl', 'w') as f:
    for row in tqdm.tqdm(data[:]):
        row = json.loads(row)

        if row['label'] != 'NOT ENOUGH INFO':
            continue

        label = row['label']
        claim = row['claim']
        evidence_lines = row['evidence']

        evidences = []
        for evidence_line in evidence_lines:            
            evidences.append(evidence_line[2])
        
        row_processed = {'label': label, 'claim': claim, 'evidence': evidences}
        
        f.write(json.dumps(row_processed) + '\n')

print(evidence_missing)