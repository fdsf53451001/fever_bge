"""
@file   : run_sentencebert.py
@author : xiaolu
@email  : luxiaonlp@163.com
@time   : 2022-02-23
"""
import os
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from config_cls import set_args
from model_cls import Model
from torch.utils.data import DataLoader
from utils import compute_corrcoef, l2_normalize, compute_pearsonr

from transformers.models.bert import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from data_helper import load_data, SentDataSet, collate_func, convert_token_id
import json

args = set_args()

def split_lines(lines):
    lines = lines.split('\n')  # 以'\n'分割字符串成行
    result = []

    for line_number, line in enumerate(lines):
        if line:
            parts = line.split('\t', 1)[1]  # 以'\t'分割行成部分
            # result.append((line_number, parts))
            result.append(parts)

    return result

def get_sentence_cls():
    dataset = open(args.test_data_path, 'r', encoding='utf-8')
    save = open(args.save_file, 'a+', encoding='utf-8')
    # model.load_state_dict(torch.load(f"./checkpoint/sentenceBert_cls_1epoch.ckpt"))
    model.load_state_dict(torch.load(f"./checkpoint/SentenceBert_3cls.ckpt"))
    model.eval()
    for data in tqdm(dataset, desc='getting similar sentence...'):
        data = eval(data)
        claimId = data['id']
        claim = data['claim']
        evidences = data['evidences'][:2]
        sent2sim = {}
        for ev_sent in evidences:
            lines = split_lines(ev_sent[1])
            for line in lines:
                if line in sent2sim or len(line)<5:
                    continue
                sent2sim[line] = classify(claim, line, model, tokenizer)
        sent2sim = list(sent2sim.items())
        # ev_sent = [s[0] for s in sent2sim if s[1] == 0 or s[1]==2]
        ev_sent = [s[0] for s in sent2sim if s[1] == 1]
        # print(ev_sent)
        data = json.dumps({'claimId': claimId, 'claim': claim, 'evidences': ev_sent}, ensure_ascii=False)
        save.write(data + "\n")
    save.close()

def classify(sent1, sent2, model, tokenizer):
    model.eval()
    s1_input_ids, s1_mask, s1_segment_id = convert_token_id(sent1, tokenizer)
    s2_input_ids, s2_mask, s2_segment_id = convert_token_id(sent2, tokenizer)

    if torch.cuda.is_available():
        s1_input_ids, s2_input_ids = s1_input_ids.cuda(), s2_input_ids.cuda()

    with torch.no_grad():
        logits = model(s1_input_ids, s2_input_ids, encoder_type='last-avg')
    logits = torch.argmax(logits, dim=-1)# -1 & 1 are the same specify column

    return logits.item()


def evaluate(model):
    model.eval()
    # 语料向量化
    all_a_vecs, all_b_vecs = [], []
    all_labels = []
    for step, batch in tqdm(enumerate(val_dataloader)):
        if torch.cuda.is_available():
            batch = (t.cuda() for t in batch)
        s1_input_ids, s2_input_ids, label_id = batch
        if torch.cuda.is_available():
            s1_input_ids, s2_input_ids, label_id = s1_input_ids.cuda(), s2_input_ids.cuda(), label_id.cuda()
        with torch.no_grad():
            s1_embeddings = model.encode(s1_input_ids, encoder_type='last-avg')
            s2_embeddings = model.encode(s2_input_ids, encoder_type='last-avg')
            s1_embeddings = s1_embeddings.cpu().numpy()
            s2_embeddings = s2_embeddings.cpu().numpy()
            label_id = label_id.cpu().numpy()

            all_a_vecs.extend(s1_embeddings)
            all_b_vecs.extend(s2_embeddings)
            all_labels.extend(label_id)

    all_a_vecs = np.array(all_a_vecs)
    all_b_vecs = np.array(all_b_vecs)
    all_labels = np.array(all_labels)

    a_vecs = l2_normalize(all_a_vecs)
    b_vecs = l2_normalize(all_b_vecs)
    sims = (a_vecs * b_vecs).sum(axis=1)
    corrcoef = compute_corrcoef(all_labels, sims)
    pearsonr = compute_pearsonr(all_labels, sims)
    return corrcoef, pearsonr

# remember to change 90 line in "data_helper.py" to tensor.long
if __name__ == '__main__':
    os.makedirs(args.output_dir, exist_ok=True)
    # tokenizer = BertTokenizer.from_pretrained(args.bert_pretrain_path)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrain_path)

    # label = {0,1,2} => {entailment, neutral, contradiction}, {0,1} => {unrelative, relative}
    train_df = load_data(args.train_data_path)
    train_dataset = SentDataSet(train_df, tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, collate_fn=collate_func)

    val_df = load_data("./Data/ori_fever_sentence/dev_evi_sentenceBert.jsonl")
    val_dataset = SentDataSet(val_df[:10000], tokenizer)
    val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=args.dev_batch_size, collate_fn=collate_func)

    num_train_steps = int(len(train_dataset) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # 模型
    model = Model()
    loss_fct = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        model.cuda()
        loss_fct.cuda()
    if args.train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = 0.05 * num_train_steps
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

        for epoch in range(args.num_train_epochs):
            for step, batch in enumerate(train_dataloader):
                if torch.cuda.is_available():
                    batch = (t.cuda() for t in batch)
                s1_input_ids, s2_input_ids, label_id = batch
                if torch.cuda.is_available():
                    s1_input_ids, s2_input_ids, label_id = s1_input_ids.cuda(), s2_input_ids.cuda(), label_id.cuda()

                logits = model(s1_input_ids, s2_input_ids, encoder_type='last-avg')
                loss = loss_fct(logits, label_id)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if step % 300 == 0:
                    print('Epoch:{}, Step:{}, Loss:{:10f}'.format(epoch, step, loss))

                loss.backward()
                # nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2)   # 是否进行梯度裁剪

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # 一轮跑完 进行eval
            corrcoef, pearsonr = evaluate(model)
            print('epoch:{}, spearmanr:{:10f}, pearsonr:{:10f}'.format(epoch, corrcoef, pearsonr))
            model.train()

            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            torch.save(model_to_save.state_dict(), f'./checkpoint/SentenceBert_3cls.ckpt')
        # Evaluate
    if args.eval:
        # Get similar sentence
        cossim = get_sentence_cls()