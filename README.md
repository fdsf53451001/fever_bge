# fever_bge
use bge embedding and reranker to do sentence retrivel for fever dataset

# architecture
![fact_check_arch drawio](https://github.com/fdsf53451001/fever_bge/assets/35889113/d5546547-d914-42ff-8921-13e596572569)

# how to generate top-k evidences
1. load wiki pages into chroma db
2. generate top100 evidence by cosine similarity check for bge embedding
3. get the final top-k evidence by bge rerank
4. you can compute strict recall for devset data

# Evidence Retrieval
在證據檢索階段，我們過濾文件檢索獲得的文件，從中取得證據。  
分成**訓練階段**與**推論階段**，而訓練階段又分成 **Classifier、Regressor、Contrastive Learning** 三種方法。  

## 資料準備
> [下載資料集](https://drive.google.com/drive/folders/16VHsCMZGbC19Swv94V4qcE3h4t-NC4R0?usp=sharing)  
下載完成後置入 `evidence_retrieval/Data` 裡面，但 `train_contrastive.jsonl` 需放到 `FlagEmbedding/Data`。  
- `ori_fever` 裡面是 FEVER 原本的資料集，在訓練前需要先處理過，包括抽取頁面的證據、標註相關或不相關，這邊先用處理過的資料直接訓練並推論看看。  
- `train_reg` 和 `train_3cls` 分別代表 Regressor 和 3 分類 Classifier(NLI task) 的訓練集，2 分類 Classifier 同樣使用 `train_reg` 訓練。  
- `train_contrastive` 是對比學習的訓練集。  
- `dev_gold_doc` 和 `dev_10doc` 都可用來測試模型的效能，差別在黃金文檔或是文檔檢索取回的 10 個文檔，它們與 `dev_gold_evi` 比對。  
> tips: 觀察不同訓練方法的 label 並比對他們的 loss function

## 執行流程
1. 訓練模型並取得 Checkpoint。  
    - 我們使用的模型架構都是 SentenceBert，基底模型皆使用 bge-large-en-v1.5。  
    - 訓練 Regressor 模型使用 `evidence_retrieval/main.py`，指令 `python main.py --train`  
    - 訓練 Classifier 模型使用 `evidence_retrieval/main_cls.py`，指令 `python main_cls.py --train`  
    - 訓練 Contrastive Learning 模型須使用 `FlagEmbedding/run.sh`，指令 `bash run.sh`  
    > tips: 訓練時經常遇到硬體限制問題，嘗試調整 batch_size 等參數，他們都在 `config.py` 或 `config_cls.py`，對比學習的在 `run.sh`  
    > [trainer parameters](https://huggingface.co/docs/transformers/main_classes/trainer)  
2. 使用訓練好的模型推論  
    - 我們需要先過濾文檔並計算出結果後，再比較與標準答案的差距，評量指標主要看 F1。  
    - 使用 Regressor 模型使用 `evidence_retrieval/main.py`，指令 `python main.py --eval`  
    - 使用 Classifier 模型使用 `evidence_retrieval/main_cls.py`，指令 `python main_cls.py --eval`  
    - 使用 Contrastive Learning 模型須使用 `evidence_retrieval/bge_embedding.py`，指令 `bash run.sh`  
        - 需要把 checkpoint 的資料集放置到 `evidence_retrieval/checkpoint` 資料夾中，且訓練好的 checkpoint 是個資料夾，將最後一次紀錄的資料夾移過來即可。  
3. 測試模型效能  
    - 到該階段只需使用 `evaluate.py`，記得到程式裡面改路徑，把 `YOUR_RESULT.jsonl` 改成你的輸出結果。  
    > tips: 這邊不考慮沒有黃金證據的情況。  