"""
Reference : https://aclanthology.org/W19-6120.pdf#page10
"""
import gc
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from src.utils import polarity_map
from src.en_dataloader import read_train_xml, read_test_xml
from src.ko_dataloader import read_train_dataset, read_test_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW


model_name = 'bert-base-multilingual-cased'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
source = {
    'en': [read_train_xml, read_test_xml, 'bert_token_cls_epoch_4_loss_0.06911052018404007.pt'],
    'ko': [read_train_dataset, read_test_dataset, 'bert_token_cls_epoch_4_loss_0.07896306365728378.pt']
}


class SentimentalPolarityDataset(Dataset):
    def __init__(self, extractor: bool,  lang: str):
        self.extractor = extractor
        self.lang = lang
        self.data = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': []
        }
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self._load_from_text()

    def _load_from_text(self):
        rows = source[self.lang][0]()
        for text, sentiments in rows:
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
        return {k: v[index].to(device) for k, v in self.data.items()}


def train_aspect_sentimental_classifier(epochs=5, extractor=False, lang='en'):
    dataset = SentimentalPolarityDataset(extractor=extractor, lang=lang)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    num_labels = 2 if extractor else 4
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
    optim = AdamW(model.parameters(), lr=5e-6)
    model.to(device)
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


def merge_tokens(filtered_tokens: np.ndarray, filtered_result: np.ndarray):

    def extract_merger(start_idx: int, end_idx: int = None):
        tokens = filtered_tokens[start_idx:end_idx]
        sentiments = filtered_result[start_idx:end_idx]
        sentiment = Counter(sentiments).most_common(n=1)
        sentiment = sentiment[0][0] if len(sentiment) else None
        merged_word = ''.join([token.replace('##', '') for token in tokens])
        merged_word = None if merged_word == '' else merged_word
        if None not in [merged_word, sentiment]:
            print(merged_word, sentiment)

    sep = [i for i, token in enumerate(filtered_tokens) if not token.startswith('##')]
    last_idx = 0
    for i, curr_idx in enumerate(sep[:-1]):
        next_idx = sep[i+1]
        last_idx = next_idx
        extract_merger(start_idx=curr_idx, end_idx=next_idx)

    # for remains
    extract_merger(start_idx=last_idx)

        
def evaluate_aspect_sentimental_classifier(model_path: str, lang='en'):
    model = BertForTokenClassification.from_pretrained(model_path)
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    polarity_map_reverse = {v: k for k, v in polarity_map.items()}
    sentences = source[lang][1]()
    for sentence in sentences:
        inputs = tokenizer.encode_plus(sentence, return_tensors='pt', padding='max_length')
        input_ids = inputs.get('input_ids')
        tokens = np.array([vocab.get(int(input_id)) for input_id in input_ids[0]])
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        result = torch.argmax(probs, dim=-1)[0]
        result = np.array(result)
        filtered_tokens = tokens[(tokens != '[CLS]') & (tokens != '[UNK]') & (tokens != '[UNK]') & (tokens != '[PAD]') & (result != 0)]
        filtered_result = result[(tokens != '[CLS]') & (tokens != '[UNK]') & (tokens != '[UNK]') & (tokens != '[PAD]') & (result != 0)]
        filtered_result = np.array(list(map(lambda elem: polarity_map_reverse.get(elem), filtered_result)))
        print('\n', sentence)
        merge_tokens(filtered_tokens, filtered_result)


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    lang = 'ko'
    clear_gpu_memory()

    # Fine-tuning
    # train_aspect_sentimental_classifier(lang='lang)

    # Evaluate
    prefix = '..'
    model_path = f'{prefix}/{source[lang][2]}'
    print(model_path)
    evaluate_aspect_sentimental_classifier(model_path, lang=lang)
