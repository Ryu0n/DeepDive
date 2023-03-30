"""
Reference : https://aclanthology.org/W19-6120.pdf#page10
"""
import os
import sys
import gc
import torch
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from itertools import chain
from tqdm import tqdm
from transformers import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
from train.utils import polarity_map, Arguments
from train.ko_dataloader import read_train_dataset, read_test_dataset

polarity_map_reverse = {v: k for k, v in polarity_map.items()}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
source = {
    'ko': [read_train_dataset, read_test_dataset]
}


class SentimentalPolarityDataset(Dataset):
    def __init__(self):
        self.data = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': []
        }
        self.tokenizer = Arguments.instance().tokenizer
        self._load_from_text()

    def _load_from_text(self):
        rows = source['ko'][0]()
        for text, sentiments in rows:
            output = self.tokenizer.encode_plus(text, return_tensors='pt', padding='max_length', truncation=True)
            for k, v in output.items():
                v = torch.squeeze(v)
                self.data.get(k).append(v)
            sentiments = torch.squeeze(torch.tensor(sentiments, dtype=torch.long))
            self.data.get('labels').append(sentiments)

    def __len__(self):
        return len(self.data.get('labels'))

    def __getitem__(self, index):
        return {k: v[index].to(device) for k, v in self.data.items()}


def train_aspect_sentimental_classifier(epochs=5):
    model_path = Arguments.instance().model_path
    tokenizer_name = Arguments.instance().tokenizer_name
    model_path = tokenizer_name if model_path is None else model_path
    model_class = Arguments.instance().model_class
    dataset = SentimentalPolarityDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = model_class.from_pretrained(model_path, num_labels=4)
    optim = AdamW(model.parameters(), lr=2e-5)
    model.to(device)
    model.train()
    lowest_loss, model_path = 99.0, None
    if not os.path.exists("./models"):
        os.mkdir('./models')

    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        loss_val = None
        total_loss = 0
        for batch in loop:
            optim.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 6)
            optim.step()
            loss_val = round(loss.item(), 3)
            total_loss += loss_val
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss_val)
        m = ''
        if 'bert' in tokenizer_name.lower():
            m = 'bert'
        elif 'electra' in tokenizer_name.lower():
            m = 'electra'
        avg_train_loss = total_loss / len(dataloader)
        checkpoint = f'models/absa_{m}_token_cls_epoch_{epoch}_loss_{round(avg_train_loss, 3)}.pt'
        if avg_train_loss < lowest_loss:
            model_path = checkpoint
            lowest_loss = loss_val
        model.save_pretrained(checkpoint)
    Arguments.instance().model_path = model_path


def post_process(true_sentiments: np.ndarray, pred_sentiments: np.ndarray):

    filtered_true_sentiments = [sentiment[sentiment != -100] for sentiment in true_sentiments]
    filtered_pred_sentiments = [[p for t, p in zip(true_sentiment, pred_sentiment) if t != -100] for true_sentiment, pred_sentiment, in zip(true_sentiments, pred_sentiments)]

    filtered_true_sentiments = np.array(list(chain(*filtered_true_sentiments)))
    filtered_pred_sentiments = np.array(list(chain(*filtered_pred_sentiments)))

    return filtered_true_sentiments, filtered_pred_sentiments


def show_merged_sentence(sentence: str, result: np.ndarray):
    tokenizer = Arguments.instance().tokenizer
    inputs = tokenizer.encode_plus(sentence,
                                   return_tensors='pt',
                                   padding='max_length',
                                   truncation=True,
                                   return_offsets_mapping=True)
    tokens, offsets = inputs.tokens(), inputs.get('offset_mapping').tolist()[0]
    chunks, chunk, curr_sentiment = list(), list(), result[1]  # 0 is [CLS]
    for token, sentiment, offset in zip(tokens, result, offsets):
        if token in tokenizer.special_tokens_map.values():
            continue
        if curr_sentiment != sentiment:
            chunk.append(curr_sentiment)
            chunks.append(chunk.copy())
            chunk = list()
        chunk.append((token, offset))
        curr_sentiment = sentiment
    if len(chunk) > 0:
        chunk.append(curr_sentiment)
        chunks.append(chunk.copy())

    end_index, merged_token_dicts = 0, list()
    for chunk in chunks:
        merged_token = ''
        span_indices = list()
        tokens, sentiment = chunk[:-1], chunk[-1]
        for token, offset in tokens:
            start, end = offset
            gap = ' '*(start - end_index)
            merged_token += f'{gap}{token}'
            span_indices.extend(offset)
            end_index = end
        if len(span_indices) == 0:
            continue
        span_indices = [min(span_indices), max(span_indices)]
        merged_token_dict = {
            'merged_token': merged_token.strip().replace('##', ''),
            'span_indices': span_indices,
            'sentiment': sentiment
        }
        # print(merged_token_dict)
        merged_token_dicts.append(merged_token_dict)

    merged_sentence, end_index = '', 0
    for merged_token_dict in merged_token_dicts:
        start, end = merged_token_dict.get('span_indices')
        gap = ' '*(start - end_index)
        if merged_token_dict.get('sentiment') != 'unrelated':
            merged_token = f'[{merged_token_dict.get("merged_token")} : {merged_token_dict.get("sentiment")}]'
        else:
            merged_token = merged_token_dict.get("merged_token")
        merged_sentence += f'{gap}{merged_token}'
        end_index = end
    print(merged_sentence)
    return {
        "origin_sentence": sentence,
        "merged_sentence": merged_sentence,
        "spans": merged_token_dicts
    }


def evaluate_aspect_sentimental_classifier():
    model_path = Arguments.instance().model_path
    model_class = Arguments.instance().model_class
    tokenizer = Arguments.instance().tokenizer
    model = model_class.from_pretrained(model_path)
    model.eval()
    sentences = source['ko'][1]()
    pred_sentiments, true_sentiments = [], []
    for i, sentence in enumerate(sentences):
        with torch.no_grad():
            sentence, sentiments = sentence
            true_sentiments.append(sentiments)
            inputs = tokenizer.encode_plus(sentence, return_tensors='pt', padding='max_length', truncation=True)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            result = np.array(torch.argmax(probs, dim=-1)[0])
            pred_sentiments.append(result)
            result = np.array(list(map(lambda elem: polarity_map_reverse.get(elem), result)))
            print('\n', sentence)
            show_merged_sentence(sentence, result)

    pred_sentiments = np.array(pred_sentiments)
    true_sentiments = np.array(true_sentiments)
    true_sentiments, pred_sentiments = post_process(true_sentiments, pred_sentiments)

    report = classification_report(y_true=true_sentiments.flatten(),
                                   y_pred=pred_sentiments.flatten(),
                                   target_names=list(polarity_map.keys()),
                                   labels=list(polarity_map.values()))
    with open('report.txt', 'w') as f:
            f.write(report)
            print(report)


def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    clear_gpu_memory()

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', required=True)
    parser.add_argument('--eval', required=True, help='if --eval is True, you must pass --model_path')
    parser.add_argument('--model_path', required=False, default=None)
    parser.add_argument('--tokenizer_name', required=True)

    args = parser.parse_args()
    args = Arguments.instance(args)

    # Fine-tuning
    if args.args.train:
        train_aspect_sentimental_classifier()

    # Evaluate
    if args.args.eval and args.args.model_path:
        evaluate_aspect_sentimental_classifier()
