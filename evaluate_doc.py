from tqdm import tqdm


pred_file = open('result/devset_evidence_rerank_10.jsonl', 'r', encoding='utf-8')
gold_file = open('dataset/share_task_dev.jsonl', 'r', encoding='utf-8')


precision = []
recall = []
f1 = []
len_gold = []
len_pred = []
for pred, gold in tqdm(zip(pred_file, gold_file), desc='Evaluating'):
    pred_evidences = eval(pred)['evidences']
    gold_evidences = eval(gold)['documents']
    tp = 0
    for pred in pred_evidences:
        if pred in gold_evidences:
            tp += 1
    if len(pred_evidences)==0 and len(gold_evidences)!=0:
        each_precision = 0
        each_recall = 0
        each_f1 = 0
    elif len(pred_evidences)!=0 and len(gold_evidences)==0:
        each_precision = 0
        each_recall = 0
        each_f1 = 0
        # each_precision = 1
        # each_recall = 1
        # each_f1 = 1
    elif len(pred_evidences)==0 and len(gold_evidences)==0:
        each_precision = 1
        each_recall = 1
        each_f1 = 1
    else:
        each_precision = tp / len(pred_evidences)
        each_recall = tp / len(gold_evidences)
        if each_precision + each_recall == 0:
            each_f1 = 0
        else:
            each_f1 = 2 * each_precision * each_recall / (each_precision + each_recall)
    precision.append(each_precision)
    recall.append(each_recall)
    f1.append(each_f1)
    len_gold.append(len(gold_evidences))
    len_pred.append(len(pred_evidences))
    # print(len(pred_evidences))
    # if len(gold_evidences) > 5:
    #     print(len(gold_evidences))
    # print("-"*10)


precision = sum(precision) / len(precision)
recall = sum(recall) / len(recall)
f1 = sum(f1) / len(f1)
print("Precision: {:.2%}".format(precision))
print("   Recall: {:.2%}".format(recall))
print("       F1: {:.2%}".format(f1))
print("len_pred:",sum(len_pred) / len(len_pred))
print("len_gold:",sum(len_gold) / len(len_gold))