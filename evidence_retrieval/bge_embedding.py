from sentence_transformers import SentenceTransformer, util
from config import set_args
from tqdm import tqdm
import json

args = set_args()

def get_similar_sentence(model):
    dataset = open(args.test_data_path, 'r', encoding='utf-8')
    save = open('./output/SentenceBert_contrastive.json', 'a+', encoding='utf-8')
    for data in tqdm(dataset): #, desc='getting similar sentence...'
        data = eval(data)
        claimId = data['claimId']
        claim = data['claim']
        evidences = data['evidences']
        label = data['label']
        sent2sim = {}
        for ev_sent in evidences:
            if ev_sent in sent2sim:
                continue
            sent2sim[ev_sent] = cosSimilarity(claim, ev_sent, model)
        sent2sim = list(sent2sim.items())
        sent2sim.sort(key=lambda s: s[1], reverse=True)
        # ev_sent = [s[0] for s in sent2sim if len(s[0]) > 0 and s[1] > 0.45] # Use when threshold
        # ev_sent = [item[0] for item in ev_sent] # Use when top_k
        data = json.dumps({'claimId': claimId, 'claim': claim, 'evidences': sent2sim, 'label': label}, ensure_ascii=False)
        save.write(data + "\n")
    save.close()

def cosSimilarity(sent1, sent2, model):
    # if torch.cuda.is_available():
    #     s1_input_ids, s2_input_ids = s1_input_ids.cuda(), s2_input_ids.cuda()
    embeddings_1 = model.encode(sent1, normalize_embeddings=True)
    embeddings_2 = model.encode(sent2, normalize_embeddings=True)
    cos_sim = util.cos_sim(embeddings_1, embeddings_2)
    
    return cos_sim.item()

if __name__ == '__main__':
    # model = SentenceTransformer('BAAI/bge-large-zh-v1.5')
    model = SentenceTransformer('./checkpoint/contrastive_ckpt')
    get_similar_sentence(model)