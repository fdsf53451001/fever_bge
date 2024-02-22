import copy
import os
import random
import torch
from BertFineTune import BertFineTune
from datasets import load_dataset,Dataset, concatenate_datasets,DatasetDict
from Prompt.HardPrompt import HardPrompt
from Hyperparameters import Hyperparameters
from Prompt.P_tuning_v1 import P_tuning_v1
from tool.Evaluator import Evaluator
from tool.LeaningCurveDrawer import LeaningCurveDrawer
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
hyperparameters = Hyperparameters()

# Load Data
def load_data():
    raw_dataset = DatasetDict()
    df = pd.read_json("verify/train_processed.jsonl", lines=True)
    dataset = Dataset.from_pandas(df)
    # dataset = dataset.shuffle(seed=42)
    dataset_tmp1 = dataset.train_test_split(test_size=0.2)  # Split into train 80% and the rest 20%
    dataset_tmp2 = dataset_tmp1['test'].train_test_split(test_size=0.5)  # Split the rest 20% into validation and test sets

    raw_dataset['train'] = dataset_tmp1['train']
    raw_dataset['validation'] = dataset_tmp2['train']
    raw_dataset['test'] = dataset_tmp2['test']

    # raw_dataset['train'] = load_dataset('json',data_files=hyperparameters.train_path,split="train")
    # raw_dataset['validation'] = load_dataset('json',data_files=hyperparameters.val_path,split="train")
    # raw_dataset['test'] = load_dataset('json',data_files=hyperparameters.test_path,split="train")
    
    hyperparameters.train_size = len(raw_dataset['train'])
    hyperparameters.val_size = len(raw_dataset['validation'])
    hyperparameters.test_size = len(raw_dataset['test'])
    return raw_dataset

def load_data_testset_only():
    raw_dataset = DatasetDict()
    testset_df = pd.read_json("verify/testset_evidences_3doc_nodiff.jsonl", lines=True)
    dataset = Dataset.from_pandas(testset_df)
    # dataset = dataset.shuffle(seed=42)
    raw_dataset['test'] = dataset
    hyperparameters.test_size = len(raw_dataset['test'])
    return raw_dataset, testset_df

def data_augmentation(datasets, traget_label, num):
    
    label_mapping_datas = []
    for data in datasets["train"]:
        if data['label'] == traget_label:
            label_mapping_datas.append(data)
    # 隨機取1000筆
    random.seed(1234)
    new_datas  = random.sample(label_mapping_datas, num)

    new_dataset={'claim': [], 'label': [], 'evidence': [], 'id': [], 'verifiable': [], 'original_id': []}
    for i in range(1000):
        new_data = copy.deepcopy(new_datas[i])
        new_data["evidence"] = [[""]]
        new_data["label"] = "NOT ENOUGH INFO"
        # 加入目標
        for k,v in new_data.items():
            new_dataset[k].append(v)
    new_dataset = Dataset.from_dict(new_dataset)
    datasets["train"] = concatenate_datasets([datasets["train"],new_dataset])
    hyperparameters.train_size = len(datasets['train'])
"""
prompt_tune模型
"""
def prompt():
    # "hard_prompt"、"p_tuning"
    method = "p_tuning"
    # "bert"、"roberta"、"t5"
    model_name = "bert"
    # "bert-base-uncased"、"roberta-base"、"t5-base"、"bert-large-uncased"
    model_path = "bert-base-uncased"
    # ""、"_large"、"_freeze"
    ps="_Promptbert_out5"
    method_model = f"{method}{ps}_{model_name}"

    if method == "hard_prompt":
        prompt = HardPrompt(model_name, model_path)
    elif method == "p_tuning":
        prompt = P_tuning_v1(model_name, model_path)
    # Load Data
    datasets = load_data()

    # 設定參數
    prompt.training_args.use_cuda = hyperparameters.use_cuda
    prompt.training_args.learn_rate = hyperparameters.learn_rate
    prompt.training_args.epoch = hyperparameters.epoch
    prompt.training_args.weight_decay = hyperparameters.weight_decay
    prompt.training_args.batch_size = hyperparameters.batch_size
    prompt.training_args.max_seq_length = hyperparameters.max_seq_length
    prompt.training_args.staging_point = f'./temp/{method_model}.pth'

    # 印出參數
    hyperparameters.print(method_model)

    # 資料前處理
    prompt.datasets_process(datasets)

    # 增加SUPPORTS 1000筆並去除evidence
    # data_augmentation(datasets,"SUPPORTS",1000)

    # 使用peft
    # peft_config = LoraConfig(
    # task_type=TaskType.TOKEN_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
    # )
    # prompt.plm = get_peft_model(prompt.plm, peft_config)

    # 是否載入之前模型
    # prompt.model.load_state_dict(torch.load(prompt.training_args.staging_point))
    
    prompt.set_device()

    # Zero-shot
    # labels,predic = prompt.zero_shot()
    # evaluator = Evaluator(labels, predic)
    # evaluator.matrixes()
    # evaluator.accuracy()
    # evaluator.precision_recall_fscore()

    # 訓練
    prompt.train()

    # 預測
    print("--------test---------")
    labels,predic = prompt.label_predict()
    evaluator = Evaluator(labels, predic)
    evaluator.matrixes()
    evaluator.accuracy()
    evaluator.precision_recall_fscore()
    evaluator.save(f"output/{method_model}")

"""
fine_tune模型
"""
def fine_tune():
    # Load Data
    datasets = load_data()

    # "bert-base-uncased"、"roberta-base"、"bert-large-uncased"
    model_path = "roberta-base"
    method_model = f"fineTune_{model_path}"

    # BertFineTune
    bertFineTune = BertFineTune(model_path)

    # 設定參數
    bertFineTune.training_args.use_cuda = hyperparameters.use_cuda
    bertFineTune.training_args.learn_rate = hyperparameters.learn_rate
    bertFineTune.training_args.epoch = hyperparameters.epoch
    bertFineTune.training_args.weight_decay = hyperparameters.weight_decay
    bertFineTune.training_args.batch_size = hyperparameters.batch_size
    bertFineTune.training_args.max_seq_length = hyperparameters.max_seq_length
    # 設定trainArgument
    bertFineTune.set_trainArgument()
    # 印出參數
    hyperparameters.print(method_model)
    # 載入暫存點
    bertFineTune.trainingArguments.resume_from_checkpoint = "./temp/checkpoint-5000"
    
    # 資料前處理
    bertFineTune.datasets_process(datasets)
    # 設定
    bertFineTune.set_device()
    # 訓練
    bertFineTune.set_trainer()
    bertFineTune.train()
    bertFineTune.trainer.save_model(f"temp/{method_model}.ckpt")

    # 預測
    labels,predic = bertFineTune.label_predict()
    evaluator = Evaluator(labels, predic)
    evaluator.matrixes()
    evaluator.accuracy()
    evaluator.precision_recall_fscore()
    evaluator.save(f"output/{method_model}")


def test():
    # Load Data
    datasets, testset_df = load_data_testset_only()

    # "bert-base-uncased"、"roberta-base"、"bert-large-uncased"
    model_path = "temp/fineTune_roberta-base.ckpt_1tn"
    method_model = f"fineTune_{model_path}"

    # BertFineTune
    bertFineTune = BertFineTune(model_path)

    # 設定參數
    bertFineTune.training_args.use_cuda = hyperparameters.use_cuda
    bertFineTune.training_args.learn_rate = hyperparameters.learn_rate
    bertFineTune.training_args.epoch = hyperparameters.epoch
    bertFineTune.training_args.weight_decay = hyperparameters.weight_decay
    bertFineTune.training_args.batch_size = hyperparameters.batch_size
    bertFineTune.training_args.max_seq_length = hyperparameters.max_seq_length
    # 設定trainArgument
    bertFineTune.set_trainArgument()
    # 印出參數
    hyperparameters.print(method_model)
    # 載入暫存點
    bertFineTune.trainingArguments.resume_from_checkpoint = model_path
    
    # 資料前處理
    bertFineTune.datasets_process(datasets, no_label=True)
    # 設定
    bertFineTune.set_device()

    label_mapping = {0:"SUPPORTS", 1:"REFUTES", 2:"NOT ENOUGH INFO"}
    predictions, scores = bertFineTune.predictions()
    for i in range(len(predictions)):
        if testset_df.loc[i, 'predicted_label'] == "":
            testset_df.loc[i, 'predicted_label'] = label_mapping[int(predictions[i])]
        testset_df.loc[i, 'predicted_score'] = scores[i][predictions[i]]
    testset_df.to_json("output/testset_with_prediction_3doc_nodiff.jsonl", orient='records', lines=True)


if __name__ == '__main__':
    # fine_tune()
    test()
    # prompt()
    # learn_curve()