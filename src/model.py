"""
Reference : https://aclanthology.org/W19-6120.pdf#page10
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import gc
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter
from sklearn.metrics import f1_score
from src.utils import polarity_map, Arguments, is_entity
from src.en_dataloader import read_train_xml, read_test_xml
from src.ko_dataloader import read_train_dataset, read_test_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW


device = 'cuda' if torch.cuda.is_available() else 'cpu'
source = {
    'en': [read_train_xml, read_test_xml],
    'ko': [read_train_dataset, read_test_dataset]
}


class SentimentalPolarityDataset(Dataset):
    def __init__(self, extractor: bool):
        self.extractor = extractor
        self.lang = Arguments.instance().args.lang
        self.model_class = Arguments.instance().model_class
        self.tokenizer_class = Arguments.instance().tokenizer_class
        self.model_path = Arguments.instance().args.model_path
        self.tokenizer_name = Arguments.instance().args.tokenizer
        self.data = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': []
        }
        self.tokenizer = self.tokenizer_class.from_pretrained(self.tokenizer_name)
        self._load_from_text()

    def _load_from_text(self):
        rows = source[self.lang][0](extractor=self.extractor)
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


def train_aspect_sentimental_classifier(epochs=5, extractor=False):
    lang = Arguments.instance().args.lang
    model_path = Arguments.instance().args.model_path
    tokenizer_name = Arguments.instance().args.tokenizer
    model_path = tokenizer_name if model_path is None else model_path
    model_class = Arguments.instance().model_class
    dataset = SentimentalPolarityDataset(extractor=extractor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    num_labels = 2 if extractor else 4
    model = model_class.from_pretrained(model_path, num_labels=num_labels)
    optim = AdamW(model.parameters(), lr=2e-5)
    model.to(device)
    model.train()
    lowest_loss, model_path = 99.0, None

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        loss_val = None
        total_loss = 0
        for batch in loop:
            optim.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optim.step()
            loss_val = round(loss.item(), 3)
            total_loss += loss_val
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss_val)
        m = ''
        if 'bert' in tokenizer_name:
            m = 'bert'
        elif 'electra' in tokenizer_name:
            m = 'electra'
        avg_train_loss = total_loss / len(dataloader)
        checkpoint = f'{lang}_{m}_token_cls_epoch_{epoch}_loss_{avg_train_loss}.pt'
        if avg_train_loss < lowest_loss:
            model_path = checkpoint
            lowest_loss = loss_val
        model.save_pretrained(checkpoint)
    Arguments.instance().args.model_path = model_path


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

    # for remainder
    extract_merger(start_idx=last_idx)

        
def evaluate_aspect_sentimental_classifier(extractor=False):
    lang = Arguments.instance().args.lang
    model_path = Arguments.instance().args.model_path
    tokenizer_name = Arguments.instance().args.tokenizer
    model_class = Arguments.instance().model_class
    tokenizer_class = Arguments.instance().tokenizer_class
    model = model_class.from_pretrained(model_path)
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    m = polarity_map if extractor else is_entity
    polarity_map_reverse = {v: k for k, v in m.items()}
    sentences = source[lang][1]()
    pred_sentiments, true_sentiments = [], []
    for sentence in sentences:
        if lang == 'ko':
            sentence, sentiments = sentence
            if extractor:
                sentiments = list(map(lambda sentiment: 1 if sentiment > 0 else 0, sentiments))
            true_sentiments.append(sentiments)
        inputs = tokenizer.encode_plus(sentence, return_tensors='pt', padding='max_length')
        input_ids = inputs.get('input_ids')
        tokens = np.array([vocab.get(int(input_id)) for input_id in input_ids[0]])
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        result = torch.argmax(probs, dim=-1)[0]
        result = np.array(result)
        pred_sentiments.append(result)
        filtered_tokens = tokens[(tokens != '[CLS]') & (tokens != '[UNK]') & (tokens != '[SEP]') & (tokens != '[PAD]') & (result != 0)]
        filtered_result = result[(tokens != '[CLS]') & (tokens != '[UNK]') & (tokens != '[SEP]') & (tokens != '[PAD]') & (result != 0)]
        filtered_result = np.array(list(map(lambda elem: polarity_map_reverse.get(elem), filtered_result)))
        print('\n', sentence)
        merge_tokens(filtered_tokens, filtered_result)

    pred_sentiments = np.array(pred_sentiments)
    true_sentiments = np.array(true_sentiments)

    # https://stackoverflow.com/questions/33326704/scikit-learn-calculate-f1-in-multilabel-classification
    if lang == 'ko':
        m = MultiLabelBinarizer().fit(true_sentiments)
        f1 = f1_score(m.transform(true_sentiments), m.transform(pred_sentiments), average='macro')
        print('f1 score : ', f1)


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    clear_gpu_memory()

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', required=True)
    parser.add_argument('--eval', required=True, help='if --eval is True, you must pass --model_path')
    parser.add_argument('--lang', required=False, default='en')
    parser.add_argument('--model_path', required=False, default=None)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--extractor', required=True)

    args = parser.parse_args()
    args = Arguments.instance(args)

    # Fine-tuning
    if args.args.train:
        train_aspect_sentimental_classifier(extractor=args.args.extractor)

    # Evaluate
    if args.args.eval and args.args.model_path:
        evaluate_aspect_sentimental_classifier(extractor=args.args.extractor)
