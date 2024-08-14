import json
import tqdm
from utils.cal_embedding_bge_en import calculate_docs_embedding, get_embeddings
from utils.cal_embedding_bge_zh import calculate_docs_embedding_zh, get_embeddings_zh

from langchain.vectorstores.chroma import Chroma
import chromadb
from chromadb.config import Settings

from fastapi import FastAPI, HTTPException, Body
from webETL_crawler import url_to_text
import requests

app = FastAPI()

@app.post("/get_top_k_evidences")
async def get_top_k_evidences(data: dict = Body(...)):
    try:
        url_list = data['url_list']
        claim = data['claim']
        k = data['k']
        lang = data['lang']

        verify_result_json = verify_with_crawler(url_list, claim, k, lang)
        return verify_result_json
    
    except KeyError as e:
        print(e)
        raise HTTPException(status_code=400, detail="Bad Request")

def verify_with_crawler(url_list:list, claim:str, k:int, lang='en'):
    ranked_urls = get_similar_urls_top_k(url_list, claim, k, lang)
    verify_result = everify(ranked_urls, claim, lang)
    weighted_verify_result = _rank_verify_evidences(verify_result)
    return weighted_verify_result

def _rank_verify_evidences(verify_result):
    # label 0:support, 1:refuted, 2:NEI
    weighted_verify_result = {}
    for i, verify in enumerate(verify_result):
        if verify['label'] == 2:
            continue
        else:
            weighted_verify_result['verify'] = verify['label']
            weighted_verify_result['evi_index'] = i
            break
    
    if 'verify' not in weighted_verify_result:
        weighted_verify_result['verify'] = 2

    weighted_verify_result['resources'] = verify_result
    return weighted_verify_result

def get_similar_urls_top_k(urls_list, claim, k, lang):
    texts_list = []
    for url in tqdm.tqdm(urls_list, desc='crawling'):
        text = url_to_text(url,mode='direct')
        texts_list.append(text)

    ranked_texts = get_similar_texts_top_k(texts_list, claim, k, lang)
    ranking = [texts_list.index(ranked_text[0]) for ranked_text in ranked_texts]
    ranked_urls = [urls_list[i] for i in ranking]

    return ranked_urls

def get_similar_texts_top_k(texts_list, claim, k, lang):

    if k >= len(texts_list):
        return None
    
    if lang == 'en':
        get_embeddings_function = get_embeddings
        calculate_docs_embedding_function = calculate_docs_embedding
    elif lang == 'zh':
        get_embeddings_function = get_embeddings_zh
        calculate_docs_embedding_function = calculate_docs_embedding_zh
    else:
        raise HTTPException(status_code=400, detail="Language not supported")

    client = chromadb.Client(settings=Settings(allow_reset=True))
    client.reset()
    collection = client.get_or_create_collection("evidence_extraction")
    embeddings_list = []

    for texts in tqdm.tqdm(texts_list, desc='select docs'):

        embeddings = None
        for _ in range(3): # max retry = 3
            # embeddings = calculate_embedding(text)
            embeddings = calculate_docs_embedding_function([texts])
            if embeddings:
                break

        embeddings_list.append(embeddings[0])
    
    collection.add(
        documents=texts_list,
        ids=[str(i) for i in range(len(texts_list))],
        embeddings=embeddings_list
    )

    vectorstore = Chroma(client=client, embedding_function=get_embeddings_function(), collection_name="evidence_extraction")
    documents = vectorstore.similarity_search_with_relevance_scores(claim, k=k)
    ans = []
    for (doc, score) in documents:
        ans.append((doc.page_content, score)) 
    return ans

def everify(ranked_urls, claim, lang='en') -> list:
    print('call verify API')
    verify_result = []
    for url in ranked_urls:
        if lang == 'en':
            web = requests.get(f"http://140.115.54.36/everify/?claim={claim}&url={url}")
        elif lang == 'zh':
            web = requests.get(f"http://140.115.54.36/cverify2/?claim={claim}&url={url}")
        verify_json = web.json()
        verify_json['url'] = url
        verify_result.append(verify_json)
    return verify_result

if __name__ == '__main__':
    urls = [
        'https://zh.wikipedia.org/zh-tw/%E8%94%A1%E8%8B%B1%E6%96%87',
        'https://www.president.gov.tw/Page/40',
        'https://www.president.gov.tw/Page/249'
    ]
    claim = '中華民國總統是蔡英文'
    k = 2
    print(verify_with_crawler(urls, claim, k, lang='zh'))
        
        