import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


dataset_file_names = {
    True: "data/NXNE1902008030.json_preprocess.txt",
    False: "data/SXNE1902007240.json_preprocess.txt"
}


class NerDataset(Dataset):
    def __init__(self, is_train):
        self.contents = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels": []
        }

        tokenizer = BertTokenizerFast.from_pretrained('klue/bert-base')
        dataset_file_name = dataset_file_names[is_train]

        with open(dataset_file_name, 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                sentence, label = line.split('\t')
                label = list(map(int, label.split(' ')))
                label = torch.LongTensor(label)
                inputs = tokenizer.encode_plus(sentence,
                                               return_tensors='pt',
                                               padding='max_length')
                for key, value in inputs.items():
                    value = torch.squeeze(value)
                    self.contents[key].append(value)
                self.contents["labels"].append(label)

    def __len__(self):
        return len(self.contents.get("labels"))

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.contents.items()}


def dataloader(is_train, batch_size=16, num_workers=2):
    dataset = NerDataset(is_train=is_train)
    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=num_workers)


if __name__ == "__main__":
    train_dataset = NerDataset(is_train=True)
    for e in train_dataset:
        print(e)
        break
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for batch in train_dataloader:
        print(batch)
        break
