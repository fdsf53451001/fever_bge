from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.chroma import Chroma
import chromadb
import pandas as pd
import tqdm
import sys 
import requests
import time
from typing import Tuple

from load_fever_dataset import load_fever_dataset_exclude_NEI, load_fever_dataset_include_NEI, load_fever_dataset

dev_df = load_fever_dataset_exclude_NEI('dataset/shared_task_dev.jsonl')

evidence_list = []
top_evidenct_amount = 10
for i in range(top_evidenct_amount):
    evidence_list.append('evi'+str(i+1))

result = pd.DataFrame(columns=['id','claim'].extend(evidence_list))

def get_wiki_search_api(claim:str, top_evidenct_amount:int) -> Tuple[bool, list]:
    '''
    input : claim
    outpit : list[title]
    '''
    # example
    # https://zh.wikipedia.org/w/api.php?action=query&list=search&srsearch=%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91&srlimit=3&utf8&format=json
    
    search_titles = ['' for i in range(top_evidenct_amount)]
    try:
        # do request
        url = f'https://en.wikipedia.org/w/api.php?action=query&list=search&srlimit={top_evidenct_amount}&srsearch={claim}&utf8&format=json'
        web = requests.get(url)
        result = web.json()
        # search_amount = result['continue']['sroffset']
        
        for i, row in enumerate(result['query']['search']):
            search_titles[i] = row['title']
        return (True, search_titles)
    
    except:
        return (False, search_titles)
        
        
api_max_try_count = 3

for i in tqdm.tqdm(range(len(dev_df))):
    row = dev_df.iloc[i]
    id = row['id']
    claim = row['claim']
    row = {'id':id,'claim':claim}

    # prevent rate limit
    if i % 180 == 0:
        print('sleeping...')
        time.sleep(1)

    if claim:
        for _ in range(api_max_try_count):
            documents = get_wiki_search_api(claim, top_evidenct_amount)
            if documents[0]:
                break
        for j, doc in enumerate(documents[1]):
            row['evi'+str(j+1)] = doc
    result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)

result.to_csv('result/devset_evidence_10_wikiapi.csv',index=False)




