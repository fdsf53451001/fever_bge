import json
import tqdm

REMAIN_AMOUNT = 2
pred_file = open('doc_eval/result/devset_evidence_rerank_10.jsonl', 'r', encoding='utf-8')
output_pred_file = open('doc_eval/result/devset_evidence_rerank_2.jsonl', 'w', encoding='utf-8')

for line in tqdm.tqdm(pred_file):
    data = json.loads(line)
    data['evidences'] = data['evidences'][:REMAIN_AMOUNT]
    output_pred_file.write(json.dumps(data) + '\n')

pred_file.close()
output_pred_file.close()
    