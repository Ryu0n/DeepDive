"""
Reference : https://aclanthology.org/W19-6120.pdf#page10
"""
import torch
from tqdm import tqdm
from src.dataloader import read_text
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW


model_name = 'bert-base-multilingual-cased'


class SentimentalPolarityDataset(Dataset):
    def __init__(self, extractor: bool):
        self.extractor = extractor
        self.data = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': []
        }
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self._load_from_text()

    def _load_from_text(self):
        for text, sentiments in read_text():
            output = self.tokenizer.encode_plus(text, return_tensors='pt', padding='max_length')
            for k, v in output.items():
                v = torch.squeeze(v)
                self.data.get(k).append(v)
            sentiments = torch.squeeze(torch.tensor(sentiments, dtype=torch.long))
            if self.extractor:
                sentiments[sentiments > 0] = 1
            self.data.get('labels').append(sentiments)

    def __len__(self):
        return len(self.data.get('labels'))

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.data.items()}


def train_aspect_sentimental_classifier(epochs=5):
    dataset = SentimentalPolarityDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=4)
    optim = AdamW(model.parameters(), lr=5e-6)
    model.train()

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        loss_val = None
        for batch in loop:
            optim.zero_grad()
            outputs = model(**batch)
            logits, loss = outputs.logits, outputs.loss
            loss.backward()
            optim.step()
            loss_val = loss.item()
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss_val)
        checkpoint = f'bert_token_cls_epoch_{epoch}_loss_{loss_val}.pt'
        model.save_pretrained(checkpoint)


if __name__ == "__main__":
    train_aspect_sentimental_classifier()

