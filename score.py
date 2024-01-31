import json
import tqdm
import pandas as pd

from load_fever_dataset import load_fever_dataset_exclude_NEI
# from utils.load_data import find_text_by_ids

# import ids from wiki dumps
# data = {}
# for page_index in tqdm.tqdm(range(1,110),desc='load wiki data'): #110
#     file_i = str(page_index).zfill(3)

#     file = open("fever/wiki-pages/wiki-"+file_i+".jsonl", "r")
#     page_data = file.readlines()
#     file.close()

#     for sentence_index, json_str in enumerate(page_data[1:]):
#         item = json.loads(json_str)
#         id = item.get("id")
#         if not id:
#             continue
#         data[id] = (file_i,sentence_index+1)

# def find_text_by_ids(ids):
#     if ids not in data:
#         return None
#     (file_i, sentence_index) = data[ids]
#     file = open("fever/wiki-pages/wiki-"+file_i+".jsonl", "r")
#     page_data = file.readlines()
#     file.close()
#     item = json.loads(page_data[sentence_index])
#     return item.get('text')

# print(find_text_by_ids('Murda_Beatz'))

top_evidenct_amount = 10
evidence_df = pd.read_csv('result/devset_evidence_10_wikiapi.csv')
devset_df = load_fever_dataset_exclude_NEI('dataset/shared_task_dev.jsonl')

data_length = len(evidence_df)
total_match = 0
partial_match = 0
missing_wiki_sentence = 0

for i in tqdm.tqdm(range(data_length),desc='scoring'):
    evidence_row = evidence_df.iloc[i]
    id = evidence_row['id']
    devset_row = devset_df[devset_df['id']==id]
    true_evidence = list(devset_row['evidence'])[0]
    
    evidence_amount = len(true_evidence)
    correct_predict_evidence_amount = 0
    for k in range(evidence_amount):
        true_evidence_idx = true_evidence[k][0][2]
        # true_evidence_sentence = find_text_by_ids(true_evidence_idx)
        # if not true_evidence_sentence:
        #     missing_wiki_sentence += 1
        #     continue

        for j in range(1,top_evidenct_amount+1):
            get_evidence = evidence_row['evi'+str(j)]
            # print(i,k,j,get_evidence)
            if get_evidence and not pd.isna(get_evidence):
                
                # the respond title of wiki api likes "Murda Beatz" is not the same as the title in wiki dump, replace space with _ to match the title in wiki dump
                # turn this on with the wiki api
                replace_space = True

                if replace_space:
                    get_evidence = get_evidence.replace(' ','_').replace('(','-LRB-').replace(')','-RRB-')

                # true_evidence_sentence
                if true_evidence_idx in get_evidence:
                    correct_predict_evidence_amount += 1
                    break
    if correct_predict_evidence_amount == evidence_amount:
        total_match += 1
    elif correct_predict_evidence_amount > 0:
        partial_match += 1

zero_match = data_length - total_match - partial_match

print('data_length',data_length)
print('total_match',total_match)
print('partial_match',partial_match)
print('zero_match',zero_match)
print('missing_wiki_sentence',missing_wiki_sentence)