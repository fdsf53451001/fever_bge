import torch

class Hyperparameters():
    def __init__(self):
        self.train_path = "Data/golden_evaluate/train.json"
        self.val_path = "Data/golden_evaluate/valid.json"
        self.test_path = "Data/golden_evaluate/test.json"
        # self.train_path = "output/promptbert/promptbert_out5_train.json"
        # self.val_path = "output/promptbert/promptbert_out5_valid.json"
        # self.test_path = "output/promptbert/promptbert_out5.json"
        self.train_size = None
        self.val_size = None
        self.test_size = None
        self.epoch = 30
        self.batch_size = 4
        self.learn_rate = 1e-5
        self.weight_decay = 0
        self.max_seq_length = 512
        self.use_cuda = torch.cuda.is_available()
    def print(self, title=""):
        print(title)
        print("-------------------------")
        print("TRAIN_SIZE :",self.train_size)
        print("VALIDATION_SIZE :",self.val_size)
        print("TEST_SIZE :",self.test_size)    
        print("EPOCH :",self.epoch)
        print("BATCH_SIZE :",self.batch_size)
        print("LEARN_RATE :",self.learn_rate)
        print("WEIGHT_DECAY :",self.weight_decay)
        print("MAX_SEQ_LENGTH :",self.max_seq_length)
        print("USE_CUDA :",self.use_cuda)
        print("-------------------------")

    