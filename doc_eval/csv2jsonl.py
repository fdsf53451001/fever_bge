import json
import csv
# 打開json
dataset = open('doc_eval/result/devset_evidence_v5_100.csv', 'r', encoding='utf-8')
csv_reader = csv.reader(dataset)
next(csv_reader)  # 跳过标题行
save = open('doc_eval/result/devset_evidence_v5_100.jsonl', 'a+', encoding='utf-8')
for row in csv_reader:
    id = row[0]
    claim = row[1]
    evi = row[2:102]
    write_data = json.dumps({'id': int(id), 'claim': claim, 'evidences': evi}, ensure_ascii=False)
    save.write(write_data + "\n")
save.close()