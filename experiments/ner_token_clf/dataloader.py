import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


dataset_file_names = {
    True: "static/select_star_preprocess_train.txt",
    False: "static/select_star_preprocess_test.txt"
}


class NerDataset(Dataset):
    def __init__(self, is_train, device, tokenizer):
        self.device = device
        self.contents = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels": []
        }

        self.tokenizer = tokenizer
        dataset_file_name = dataset_file_names[is_train]

        with open(dataset_file_name, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.replace('\n', '')
                sentence, label = line.split('\t')
                label = list(map(int, label.split(' ')))
                label = torch.tensor(label, dtype=torch.long)
                inputs = self.tokenizer.encode_plus(sentence,
                                                    return_tensors='pt',
                                                    padding='max_length')
                for key, value in inputs.items():
                    value = torch.squeeze(value)
                    self.contents[key].append(value)
                self.contents["labels"].append(label)

    def __len__(self):
        return len(self.contents.get("labels"))

    def __getitem__(self, index):
        return {key: value[index].to(self.device) for key, value in self.contents.items()}


def dataloader(is_train, device, tokenizer, batch_size=16):
    dataset = NerDataset(is_train=is_train, device=device, tokenizer=tokenizer)
    return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True
        )
