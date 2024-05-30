from stanfordcorenlp import StanfordCoreNLP
import json
import sys
import tqdm

sys.path.append("/home/jovyan/fever_bge/utils")
from load_data import search_ids_by_keyword

nlp = StanfordCoreNLP(r'claim_process/stanford-corenlp-full-2018-10-05')
properties = {
    'annotators': 'tokenize,ssplit,pos,lemma,ner',
    'outputFormat': 'json'
}

# sentence = "Stanford CoreNLP provides a set of natural language analysis tools written in Java."
# output = nlp.annotate(sentence, properties=properties)
# print(output)

def get_entities(text:str) -> list:
    output = nlp.annotate(text, properties=properties)
    output = json.loads(output)
    entities = []
    for sentence in output['sentences']:
        for entity in sentence['entitymentions']:
            entities.append(entity['text'])
    return entities

def search_wiki_pages(entities:list) -> list:
    search_list = []
    for entity in entities:
        entity = entity.replace(' ','_')
        search_list.extend(search_ids_by_keyword(entity, amount=3))
    return search_list

def process_dataset(input_jsonl_path='dataset/shared_task_dev.jsonl', output_jsonl_path='doc_eval/result/devset_evidence_kw_v2.jsonl'):
    input_dataset = []
    with open(input_jsonl_path, 'r') as f:
        input_dataset = f.readlines()
    
    with open(output_jsonl_path,'w') as f:
        for line in tqdm.tqdm(input_dataset):
            data = json.loads(line)
            claim = data['claim']
            entities = get_entities(claim)
            evidences = search_wiki_pages(entities)
            f.write(json.dumps({'id':data['id'], 'claim':claim, 'evidences':evidences, 'entities':entities})+'\n')

if __name__ == '__main__':
    process_dataset()
