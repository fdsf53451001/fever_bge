from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores.chroma import Chroma
import chromadb
import pandas as pd
import tqdm
import sys 

from cal_embedding_bge import get_embeddings
from fever.bge.load_fever_dataset import load_fever_dataset_exclude_NEI

# client = client = chromadb.PersistentClient(path="fever/chroma_fever")
# collection = client.get_collection("fever_full")
# vectorstore = collection.get()

dev_df = load_fever_dataset_exclude_NEI('fever/devset/shared_task_dev.jsonl')

vectorstore = Chroma("fever_full",persist_directory='fever/chroma_fever',embedding_function=get_embeddings())

evidence_list = []
top_evidenct_amount = 100
for i in range(top_evidenct_amount):
    evidence_list.append('evi'+str(i+1))

result = pd.DataFrame(columns=['id','claim'].extend(evidence_list))

# claim_prefix = 'Represent this sentence for searching relevant passages: '

for i in tqdm.tqdm(range(len(dev_df))):
    row = dev_df.iloc[i]
    id = row['id']
    claim = row['claim']
    row = {'id':id,'claim':claim}
    if claim:
        documents = vectorstore.similarity_search_with_relevance_scores(claim, k=top_evidenct_amount)
        for j, (doc, score) in enumerate(documents):
            # print('document',j,score,doc)
            row['evi'+str(j+1)] = doc.page_content
    result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)

result.to_csv('fever/bge/devset_evidence.csv',index=False)