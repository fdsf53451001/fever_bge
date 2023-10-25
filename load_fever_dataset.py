import pandas as pd

def load_fever_dataset_exclude_NEI(filepath='fever/devset/shared_task_dev.jsonl'):

    df = pd.read_json(filepath, lines=True)
    df = df[df['label'] != 'NOT ENOUGH INFO']

    return df