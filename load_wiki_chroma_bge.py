import json
import tqdm
# from cal_embedding import calculate_embedding
from cal_embedding_bge import calculate_docs_embedding
import chromadb

client = chromadb.PersistentClient(path="fever/chroma_fever")

# collection = client.create_collection("fever_full")
collection = client.get_collection("fever_full")

for i in range(1,110): #96
    file_i = str(i).zfill(3)

    file = open("wiki-pages/wiki-"+file_i+".jsonl", "r")
    data = file.readlines()
    file.close()

    ids = []
    texts = []
    embedding_list = []

    for json_str in tqdm.tqdm(data[1:],desc=str(i)):
        item = json.loads(json_str)
        id = item.get("id")
        text = item.get("text")
        lines = item.get("lines")

        if not (id and text and lines):
            continue

        embeddings = None
        for _ in range(3): # max retry = 3
            # embeddings = calculate_embedding(text)
            embeddings = calculate_docs_embedding([text])
            if embeddings:
                break

        ids.append(id)
        texts.append(text)
        embedding_list.append(embeddings[0])
    
    collection.add(
        documents=texts[:len(texts)//2],
        ids=ids[:len(texts)//2],
        embeddings=embedding_list[:len(texts)//2],
        metadatas=[{'ids':id} for id in ids[:len(texts)//2]]
    )

    collection.add(
        documents=texts[len(texts)//2:],
        ids=ids[len(texts)//2:],
        embeddings=embedding_list[len(texts)//2:],
        metadatas=[{'ids':id} for id in ids[len(texts)//2:]]
    )
        
        
        