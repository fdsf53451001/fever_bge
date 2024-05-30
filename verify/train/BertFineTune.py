from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer, EarlyStoppingCallback
import numpy as np
import sklearn.metrics as sm
import torch
from datasets import Dataset,DatasetDict
from typing import Dict
import re
import tqdm


class BertFineTune():
    def __init__(self,checkpoint, inference = False):
        
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)

        if not inference:
            self.training_args = Arguments()
            self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
    def set_trainArgument(self):
        self.trainingArguments = TrainingArguments( output_dir = "./temp", 
                                                            evaluation_strategy= "steps",
                                                            eval_steps = 5000,
                                                            save_steps = 5000, 
                                                            save_total_limit = 1,
                                                            report_to = "none", 
                                                            load_best_model_at_end = True,
                                                            metric_for_best_model = 'precision',
                                                            learning_rate = self.training_args.learn_rate,
                                                            num_train_epochs = self.training_args.epoch,
                                                            weight_decay=self.training_args.weight_decay,
                                                            per_device_train_batch_size=self.training_args.batch_size,
                                                            per_device_eval_batch_size=self.training_args.batch_size,
                                                            logging_steps=300,
                                                            )
    def set_trainer(self):
        self.trainer = Trainer(
                        self.model,
                        self.trainingArguments,
                        train_dataset=self.datasets["train"],
                        eval_dataset=self.datasets["validation"],
                        data_collator=self.data_collator,
                        tokenizer=self.tokenizer,
                        compute_metrics=self.compute_metrics,
                        callbacks = [EarlyStoppingCallback(early_stopping_patience=self.training_args.early_stopping_patience)]
                        )
        
    def datasets_process(self,datasets:DatasetDict, no_label=False):
        self.datasets = datasets
        self.datasets = self.datasets.map(self.data_process, fn_kwargs={"no_label": no_label})

        # 移除不必要特徵
        # self.datasets = self.datasets.remove_columns('id')
        # self.datasets = self.datasets.remove_columns('verifiable')
        # self.datasets = self.datasets.remove_columns('original_id')
        self.datasets = self.datasets.remove_columns('claim')
        self.datasets = self.datasets.remove_columns('evidence')
        # print(self.datasets)

        # if hasattr(self,"trainer"):
        #     self.trainer.train_dataset = self.datasets["train"]
        #     self.trainer.eval_dataset = self.datasets["validation"]
    def concatenated_text(self,example):
        sent =  example["claim"] + ' [SEP] ' + example["evidence"]
        example['sent'] = self.text_preprocessing(sent)
        return example
    
    def text_preprocessing(self, text):
        # Remove '@name'
        text = re.sub(r'(@.*?)[\s]', ' ', text)
        # Replace '&amp;' with '&'
        text = re.sub(r'&amp;', '&', text)
        # Remove trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def data_process(self, example, no_label=False):
        if not no_label:
            # 處理label
            label_mapping = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
            example["label"] = label_mapping[example["label"]]
        # 處理evidence
        temp_evidence = []
        evidence_str = ""
        # print(example['evidence'])
        for evidence in example['evidence']:
            # if len(evidence) == 1:
            #     temp_evidence.append(evidence[0])
            # else:
            #     temp_evidence.append(evidence[2])
            if evidence:
                temp_evidence.append(evidence)

        evidence_str = "[SEP]".join(temp_evidence)
        example['evidence'] = evidence_str
        # 串聯字串
        sent =  example["claim"] + ' [SEP] ' + example["evidence"]
        # encode
        encoded_sent = self.tokenizer.encode_plus(
            text=self.text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=self.training_args.max_seq_length,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,     # Return attention mask
            truncation=True
            )
        example["input_ids"] = torch.tensor(encoded_sent.get('input_ids'))
        example["attention_mask"] = torch.tensor(encoded_sent.get('attention_mask'))

        return example

    # 計算評估數值
    def compute_metrics(self,eval_preds):
        pred, labels = eval_preds
        pred = np.argmax(pred, axis=-1)
        precision, recall, macro_f1, _ = sm.precision_recall_fscore_support(y_true=labels, y_pred=pred, average='macro')
        micro_f1 = sm.f1_score(y_true=labels, y_pred=pred, average='micro')
        return {"precision": precision, "recall": recall, "macro_f1": macro_f1, "micro_f1": micro_f1}
    
    def set_device(self):
        # 使用GPU
        if self.training_args.use_cuda:
            device = torch.device("cuda") 
        else:
            device = torch.device("cpu")
        self.model.to(device)
    
    def train(self):
        self.trainer.train()

    def label_predict(self,data_part='test'):
        predictions = self.trainer.predict(self.datasets[data_part])
        return predictions.label_ids, [np.argmax(x) for x in predictions.predictions]
    
    def predictions(self):
        # 使用GPU
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)
        test_dataset = self.datasets['test']

        predictions = []
        scores=[]
        for i in tqdm.tqdm(range(len(test_dataset)), desc="Predicting"):
            input_ids = test_dataset[i]['input_ids']
            attention_mask = test_dataset[i]['attention_mask']
            with torch.no_grad():
                output = self.model(input_ids=torch.tensor(input_ids, device=device).unsqueeze(0), attention_mask=torch.tensor(attention_mask, device=device).unsqueeze(0))
                logits = output.logits
                # print(torch.softmax(logits, dim=-1))
                scores.append(torch.softmax(logits, dim=-1).tolist()[0])
                predictions.append(logits.argmax().item())
        return predictions, scores
        
# 參數預設值
class Arguments():
    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.learn_rate = 1e-4
        self.batch_size = 1
        self.epoch = 3
        self.weight_decay = 0
        self.early_stopping_patience = 3
        self.max_seq_length = 512
        self.staging_point = 'checkpoint.pth'