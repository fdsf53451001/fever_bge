import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large').to(device)
model.eval()

def rerank(query, sentences):
    pairs = [[query, sentence] for sentence in sentences]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512).to(device)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        return scores.to('cpu').tolist()

# query = 'what is panda?'
# sentences = ['hi', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']
# print(rerank(query, sentences))

evidence_df = pd.read_csv('fever/bge/devset_evidence_100.csv')
data_length = len(evidence_df)
top_evidenct_amount = 100
save_top_evidenct_amount = 10

evidence_title_list = []
for i in range(save_top_evidenct_amount):
    evidence_title_list.append('evi'+str(i+1))
result = pd.DataFrame(columns=['id','claim'].extend(evidence_title_list))

for i in tqdm.tqdm(range(data_length),desc='reranking'):
    evidence_row = evidence_df.iloc[i]
    claim = evidence_row['claim']
    evidences = [evidence_row['evi'+str(j)] for j in range(1,top_evidenct_amount+1)]
    scores = rerank(claim, evidences)

    top_evidences = []
    for j in range(save_top_evidenct_amount):
        top_evidences.append(evidences[scores.index(max(scores))])
        scores[scores.index(max(scores))] = -100
    
    row = {'id':evidence_row['id'],'claim':claim}
    for j in range(save_top_evidenct_amount):
        row['evi'+str(j+1)] = top_evidences[j]
    result = pd.concat([result, pd.DataFrame([row])], ignore_index=True)

result.to_csv('fever/bge/devset_evidence_rerank_10.csv',index=False)


