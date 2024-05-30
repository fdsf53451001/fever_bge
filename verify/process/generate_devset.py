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
with open('verify/dataset/shared_task_dev.jsonl', 'r') as f:
    data = f.readlines()

evidence_missing = 0
with open('verify/dataset/dev_processed.jsonl', 'w') as f:
    for row in tqdm.tqdm(data[:]):
        row = json.loads(row)

        label = row['label']
        claim = row['claim']
        evidence_lines = row['evidence_match_lines']

        evidences = []
        for evidence_line in evidence_lines:
            evidence_doc_split = split_lines(evidence_line[2])
            if evidence_line[2] == '' or evidence_line[1]>=len(evidence_doc_split):
                evidence_missing += 1
                continue
            line_texts = evidence_doc_split[evidence_line[1]]
            evidences.append(line_texts)
        
        row_processed = {'label': label, 'claim': claim, 'evidence': evidences}
        
        f.write(json.dumps(row_processed) + '\n')

print(evidence_missing)