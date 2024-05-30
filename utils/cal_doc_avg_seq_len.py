from load_data import data
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

def cal_average_seq_len():
    total_len = 0
    over_512_len = 0
    for key in data:
        text = data[key][2]
        input_ids = tokenizer(text)['input_ids']

        total_len += len(input_ids)
        if len(input_ids) > 512:
            over_512_len += 1

    print('average seq len:',total_len/len(data)) 
    print('over 512 len:',over_512_len)

cal_average_seq_len()