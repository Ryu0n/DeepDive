import torch
from typing import List, Tuple
from Korpora import Korpora
from torch.utils.data import Dataset, DataLoader
from utils import filter_special_characters
from transformers import BertTokenizer
from torch.cuda import is_available
from torch.backends.mps import is_available as mps_is_available


device = 'cuda' if is_available() else 'cpu'
# device = 'mps' if mps_is_available() else 'cpu'  # for silicon mac


class NSMCDataset(Dataset):
    def __init__(self, is_train: bool):

        self.contents = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "labels": []
        }

        dataset_checkpoint = 'nsmc'
        Korpora.fetch(dataset_checkpoint)
        self.corpus = Korpora.load(dataset_checkpoint)
        self.is_train = is_train
        self.tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
        self.train_corpus, self.test_corpus = self.preprocess()
        self.tokenize_sentences()

    def preprocess(self):
        train_corpus = list(map(lambda elem: (filter_special_characters(elem[0]), elem[1]),
                                     zip(self.corpus.train.texts, self.corpus.train.labels)))
        test_corpus = list(map(lambda elem: (filter_special_characters(elem[0]), elem[1]),
                                    zip(self.corpus.test.texts, self.corpus.test.labels)))
        return train_corpus, test_corpus

    def tokenize_sentences(self):
        corpus: List[Tuple[str, int]] = self.train_corpus if self.is_train else self.test_corpus
        for sentence, label in corpus:
            # 각 문장을 토크나이징
            tokenized_inputs = self.tokenizer.encode_plus(
                sentence,
                padding="max_length",
                return_tensors="pt"
            )

            # 토크나이징 결과와 라벨을 텐서로 변환하고 self.contents에 추가
            for k, v in tokenized_inputs.items():
                self.contents[k].append(v.squeeze())
            label = torch.tensor(label, dtype=torch.long)
            self.contents["labels"].append(label)

    def __len__(self):
        return len(self.contents.get("labels"))

    def __getitem__(self, index):
        return {k: v[index].to(device) for k, v in self.contents.items()}


def dataloader(is_train: bool, batch_size: int):
    dataset = NSMCDataset(is_train=is_train)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=True)
