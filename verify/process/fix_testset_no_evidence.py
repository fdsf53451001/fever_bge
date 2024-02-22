import json
from tqdm import tqdm

evis =  open('verify/dataset/created/testset_evidences_10doc.jsonl', 'r', encoding='utf-8')
vers =  open('verify/dataset/created/testset_with_prediction_count_10doc.jsonl', 'r', encoding='utf-8')
save = open('verify/dataset/created/testset_with_prediction_count_10doc_fix.jsonl', 'a+', encoding='utf-8')

for evi, ver in tqdm(zip(evis, vers)):
    evi_data = json.loads(evi)
    ver_data = json.loads(ver)
    while int(evi_data['id']) != ver_data['id']:
        write_data = json.dumps({'id': int(evi_data['id']), 'predicted_label': "NOT ENOUGH INFO", 'predicted_evidence': []}, ensure_ascii=False)
        save.write(write_data + "\n")
        evi = next(evis)
        evi_data = json.loads(evi)
    id = int(evi_data['id'])
    predicted_label = ver_data['predicted_label']
    predicted_evidence = evi_data['evidences']
    write_data = json.dumps({'id': id, 'predicted_label': predicted_label, 'predicted_evidence': predicted_evidence}, ensure_ascii=False)
    save.write(write_data + "\n")
save.close()