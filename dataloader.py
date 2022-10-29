from Korpora import Korpora
from torch.utils.data import Dataset, DataLoader


class NSMCDataset(Dataset):
    def __init__(self, is_train: bool):
        dataset_checkpoint = 'nsmc'
        Korpora.fetch(dataset_checkpoint)
        self.corpus = Korpora.load(dataset_checkpoint)
        self.is_train = is_train

    def __len__(self):
        return len(self.corpus.train if self.is_train else self.corpus.test)

    def __getitem__(self, index):
        labaled_sentence = self.corpus.train[index] if self.is_train else self.corpus.test[index]
        text, label = labaled_sentence.text, labaled_sentence.label
        return


if __name__ == "__main__":
    train_dataset = NSMCDataset(is_train=True)
    test_dataset = NSMCDataset(is_train=False)
    print(train_dataset[0])
    print(test_dataset[0])
