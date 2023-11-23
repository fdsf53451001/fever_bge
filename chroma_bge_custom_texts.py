import json
import tqdm
# from cal_embedding import calculate_embedding
from cal_embedding_bge import calculate_docs_embedding
from langchain.vectorstores.chroma import Chroma
import chromadb
from cal_embedding_bge import get_embeddings

def get_similar_texts_top_k(texts_list, claim, k):

    if k >= len(texts_list):
        return None

    client = chromadb.Client()
    collection = client.create_collection("evidence_extraction")
    embeddings_list = []

    for texts in tqdm.tqdm(texts_list):

        embeddings = None
        for _ in range(3): # max retry = 3
            # embeddings = calculate_embedding(text)
            embeddings = calculate_docs_embedding([texts])
            if embeddings:
                break

        embeddings_list.append(embeddings[0])
    
    collection.add(
        documents=texts_list,
        ids=[str(i) for i in range(len(texts_list))],
        embeddings=embeddings_list
    )

    vectorstore = Chroma(client=client, embedding_function=get_embeddings(), collection_name="evidence_extraction")
    documents = vectorstore.similarity_search_with_relevance_scores(claim, k=k)
    ans = []
    for (doc, score) in documents:
        ans.append((doc.metadata['ids'], score)) 
    return ans

if __name__ == '__main__':
    texts = ['“DQN之前：曾经有一段时间大家普遍认为online的TD方法只有在on-policy情况下才有收敛性保证。而off-policy方法（如Q-learning）虽然在tabular情况下可以保证收敛，但是在有linear FA情况下就无法保证收敛。而least-squares方法（如LSTD, LSPI）虽能解决off-policy和linear FA下的收敛性问题，但计算复杂度比较高。Sutton和Maei等人提出的GTD（Gradient-TD）系方法（如Greedy-GQ）解决了off-policy，FA，online算法的收敛性问题。它最小化目标MSPBE（mean-squared projected Bellman error），通过SGD优化求解。但因为它仍是TD方法，因此仍然逃不了TD方法在上面提到的固有缺点。要避免这些缺点，一般的做法是用PG算法（如AC算法）。在Degris等人的论文《Off-Policy Actor-Critic》中将AC算法扩展到off-policy场景，提出了OffPAC算法，它是一种online, off-policy的AC算法。 OffPAC中的critic部分采用了上面提到的GTD系方法-GTD(λ)算法。','DQN之后：传统经验认为，online的RL算法在和DNN简单结合后会不稳定。主要原因是观察数据往往波动很大且前后sample相互关联。像Neural fitted Q iteration和TRPO方法通过将经验数据batch，或者像DQN中通过experience replay memory对之随机采样，这些方法有效解决了前面所说的两个问题，但是也将算法限定在了off-policy方法中。本文提出了另一种思路，即通过创建多个agent，在多个环境实例中并行且异步的执行和学习。于是，通过这种方式，在DNN下，解锁了一大批online/offline的RL算法（如Sarsa, AC, Q-learning）。”','转自： 深度增强学习（DRL）漫谈 - 从AC（Actor-Critic）到A3C（Asynchronous Advantage Actor-Critic）','Rich Sutton的书里不是给了个例子说明on policy为啥不好么。']
    claim = 'DQN之前'
    k = 3
    print(get_similar_texts_top_k(texts, claim, k))
        
        