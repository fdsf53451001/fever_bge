
from openprompt.data_utils import InputExample

class DataHelper():

    def set_gold_datasets(self,datasets):
        label_mapping = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
        temp_input_examples = []
        for data in datasets:
            temp_evidence = []
            evidence_str = ""
            for evidence in data['evidence']:
                if len(evidence) == 1:
                    temp_evidence.append(evidence[0])
                else:
                    temp_evidence.append(evidence[2])
            evidence_str = "[SEP]".join(temp_evidence)
            input_example = InputExample(text_a = evidence_str, 
                                         text_b = data['claim'], 
                                         label=label_mapping[data['label']]) 
            temp_input_examples.append(input_example)
        return temp_input_examples
    
    def set_retrieval_datasets(self,datasets):
        temp_input_examples = []
        for data in datasets:
            evidence_str = ""
            evidence_str = "[SEP]".join(data['evidences'])
            input_example = InputExample(text_a = evidence_str, 
                                         text_b = data['claim'], 
                                         label=data['label'])
            temp_input_examples.append(input_example)
        return temp_input_examples
    
    def set_test(self,claim, evidences):
        temp_input_examples = []
        for evidence in evidences:
            input_example = InputExample(text_a = evidence,
                                         text_b = claim,
                                         label=0
                                         )
            temp_input_examples.append(input_example)
        return temp_input_examples