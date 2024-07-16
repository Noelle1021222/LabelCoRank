import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,RobertaTokenizer,DebertaTokenizer,AutoTokenizer
import params


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len,meshNum):
        self.tokenizer = tokenizer
        self.data = dataframe

        self.text = dataframe[0]
        self.targets = dataframe[1]

        self.meshNum=meshNum
        self.max_len = max_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):

        text=str(self.text[index])
        target = list(map(int, self.targets[index].split()))

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation='longest_first'
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        listTag1 = [0] * self.meshNum
        for j in target:
            listTag1[j] = 1

        return {'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
                'targets': torch.tensor(listTag1, dtype=torch.float),
                }

def data_loader_onlytest(test_path,meshNum):
    test_dataset = pd.read_csv(test_path,header=None)
    print(f"TEST Dataset: {test_dataset.shape}")

    tokenizer = AutoTokenizer.from_pretrained(params.MODEL_NAME_OR_PATH)
    test_set = CustomDataset(test_dataset, tokenizer, params.MAX_LEN, meshNum)
    test_params = {'batch_size': params.TEST_BATCH_SIZE, 'shuffle': False, 'num_workers': 0}
    test_loader = DataLoader(test_set, **test_params)
    return test_loader


def data_loader(train_path, val_path,meshNum):
    train_dataset = pd.read_csv(train_path,header=None)
    valid_dataset = pd.read_csv(val_path,header=None)

    print(
        f"TRAIN Dataset: {train_dataset.shape}, "
        f"VALID Dataset: {valid_dataset.shape}",
    )

    tokenizer=AutoTokenizer.from_pretrained(params.MODEL_NAME_OR_PATH)

    training_set = CustomDataset(train_dataset, tokenizer, params.MAX_LEN, meshNum)

    validation_set = CustomDataset(valid_dataset, tokenizer, params.MAX_LEN, meshNum)

    train_params = {'batch_size': params.TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

    valid_params = {'batch_size': params.VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

    training_loader = DataLoader(training_set, **train_params)

    validation_loader = DataLoader(validation_set, **valid_params)


    return training_loader, validation_loader



