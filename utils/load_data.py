import tqdm
import json
import random
import re

# import ids from wiki dumps
data = {}
def load_wiki_pages(s_index=1, e_index=110) -> dict:
    tmp = {}
    for page_index in tqdm.tqdm(range(s_index,e_index),desc='load wiki data'): #110
        file_i = str(page_index).zfill(3)

        file = open("wiki-pages/wiki-"+file_i+".jsonl", "r")
        page_data = file.readlines()
        file.close()

        for sentence_index, json_str in enumerate(page_data[1:]):
            item = json.loads(json_str)
            id = item.get("id")
            text = item.get("text")
            if not id or not text:
                continue
            tmp[id] = (file_i,sentence_index+1,text)
    print('load wiki data done, total size:',len(tmp))
    return tmp

# load_wiki_pages(1,2)
data = load_wiki_pages(1,110)

keys_list = list(data.keys())
keys_list_lowercase = [k[0]+k[1:].lower() for k in keys_list]

def find_text_by_ids(ids):
    if ids not in data:
        return None
    (file_i, sentence_index, text) = data[ids]

    return text

def random_get_texts_exclude_ids(amount:int, ids:list):
    avaliable_ids = list(set(data.keys())-set(ids))
    random_ids_list = []
    random_texts = []
    for _ in range(amount):
        random_ids = random.choice(avaliable_ids)
        avaliable_ids.remove(random_ids)

        random_ids_list.append(random_ids)
        random_text = find_text_by_ids(random_ids)
        random_texts.append(random_text)
    return random_texts

def search_ids_by_keyword(keyword:str, amount=3) -> list:
    search_list = []
    keyword = keyword[0] + keyword[1:].lower()
    for i, key in enumerate(keys_list_lowercase):
        if keyword in key:            
            search_list.append(keys_list[i])
    return search_list[:amount]

if __name__ == '__main__':
    print(find_text_by_ids('1928_in_association_football'))