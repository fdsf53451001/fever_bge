import tqdm
import json

def find_max_score(score):
    # return the label with the highest score, if there are multiple labels with the same score, return None
    max_score = max(score.values())
    max_label = [label for label in score if score[label] == max_score]
    # way 1, take first evidence label if there are multiple labels with the same score
    if len(max_label) == 1:
        return max_label[0]
    else:
        return None
    # way 2, take first evidence label only if support score = refute score
    # if len(max_label) == 1:
    #     return max_label[0]
    # else:
    #     if score['SUPPORTS'] == score['REFUTES']:
    #         return None
    #     else:
    #         return max_label[0]

# read jsonl file
with open('verify/dataset/created/testset_with_prediction_10doc.jsonl', 'r') as f:
    data = f.readlines()

data.append('{"id": 0, "claim": "0", "evidence": "0", "predicted_label": "SUPPORTS", "predicted_score": 0}')

last_id = json.loads(data[0])['id']
last_claim = ""
labels = []
score = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
with open('verify/dataset/created/testset_with_prediction_count_10doc.jsonl', 'w') as f:
    for row in tqdm.tqdm(data[:]):
        row = json.loads(row)

        id = row['id']
        claim = row['claim']
        evidence = row['evidence']
        predicted_label = row['predicted_label']
        predicted_score = row['predicted_score']
            
        if last_id != id:
            # different id, count last score
            label = find_max_score(score)
            if not label:
                label = labels[0]
            row_processed = {'id': last_id, 'claim': last_claim, 'predicted_label': label}        
            f.write(json.dumps(row_processed) + '\n')

            # reset
            last_id = id
            labels = []
            score = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}

        # record the evidence
        labels.append(predicted_label)
        score[predicted_label] += 1
        last_claim = claim


