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

from tqdm import tqdm
from collections import Counter
from datasets import load_metric
from src.utils import polarity_map, Arguments
from src.en_dataloader import read_train_xml, read_test_xml
from src.ko_dataloader import read_train_dataset, read_test_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW


polarity_map_reverse = {v: k for k, v in polarity_map.items()}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
source = {
    'en': [read_train_xml, read_test_xml],
    'ko': [read_train_dataset, read_test_dataset]
}


class SentimentalPolarityDataset(Dataset):
    def __init__(self):
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
        self.tokenizer = self.tokenizer_class.from_pretrained(self.tokenizer_name, num_labels=4)
        self._load_from_text()

    def _load_from_text(self):
        rows = source[self.lang][0]()
        for text, sentiments in rows:
            output = self.tokenizer.encode_plus(text, return_tensors='pt', padding='max_length')
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
    lang = Arguments.instance().args.lang
    model_path = Arguments.instance().args.model_path
    tokenizer_name = Arguments.instance().args.tokenizer
    model_path = tokenizer_name if model_path is None else model_path
    model_class = Arguments.instance().model_class
    dataset = SentimentalPolarityDataset()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    model = model_class.from_pretrained(model_path, num_labels=4)
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
    filtered_tokens, filtered_result = list(filtered_tokens), list(filtered_result)
    splited_tokens, splited_sentiments = [], []
    
    # split subword tokens by word
    # 서브워드 토큰들을 단어 단위로 나눔
    while filtered_tokens:
        curr_token: str = filtered_tokens.pop(0)
        curr_sentiment: str = filtered_result.pop(0)
        if not curr_token.startswith('##'):
            # Start of subwords
            splited_tokens.append([curr_token])
            splited_sentiments.append([curr_sentiment])
        elif splited_tokens:
            # Intermediate of subwords
            splited_tokens[-1].append(curr_token)
            splited_sentiments[-1].append(curr_sentiment)

    # post process for intermediate subword tokens
    # 중간에서 감정이 존재하는 서브워드 토큰들을 제외
    sanitized_tokens, sanitized_sentiments = [], []
    for tokens, sentiments in zip(splited_tokens, splited_sentiments):
        is_not_unrelated = True
        for token, sentiment in zip(tokens, sentiments):
            if sentiment == 'unrelated':
                is_not_unrelated = False
            if is_not_unrelated is False:
                continue
            if not token.startswith('##'):
                sanitized_tokens.append(token)
                sanitized_sentiments.append([sentiment])
            else:
                sanitized_tokens[-1] += token.replace('##', '')
                sanitized_sentiments[-1].append(sentiment)

    sanitized_sentiments = map(lambda sentiment: Counter(sentiment).most_common(n=1)[0][0], sanitized_sentiments)

    for token, sentiment in zip(sanitized_tokens, sanitized_sentiments):
        if sentiment != 'unrelated':
            print(token, sentiment)


def post_process(true_sentiments: np.ndarray, pred_sentiments: np.ndarray):

    def map_to_polarity_str(filtered_sentiments: np.ndarray):
        return np.array([
            ['_'+polarity_map_reverse.get(s) for s in sentiment]
            for sentiment in filtered_sentiments
        ])

    filtered_true_sentiments = np.array(
         [sentiment[sentiment != -100]
         for sentiment in true_sentiments]
    )
    filtered_pred_sentiments = np.array(
        [np.array([p for t, p in zip(true_sentiment, pred_sentiment) if t != -100])
         for true_sentiment, pred_sentiment, in zip(true_sentiments, pred_sentiments)]
    )

    return map(map_to_polarity_str, (filtered_true_sentiments, filtered_pred_sentiments))


def evaluate_aspect_sentimental_classifier():
    lang = Arguments.instance().args.lang
    model_path = Arguments.instance().args.model_path
    tokenizer_name = Arguments.instance().args.tokenizer
    model_class = Arguments.instance().model_class
    tokenizer_class = Arguments.instance().tokenizer_class
    model = model_class.from_pretrained(model_path)
    model.eval()
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    sentences = source[lang][1]()
    pred_sentiments, true_sentiments = [], []
    for i, sentence in enumerate(sentences):
        with torch.no_grad():
            if lang == 'ko':
                sentence, sentiments = sentence
                true_sentiments.append(sentiments)
            inputs = tokenizer.encode_plus(sentence, return_tensors='pt', padding='max_length')
            input_ids = inputs.get('input_ids')
            tokens = np.array([vocab.get(int(input_id)) for input_id in input_ids[0]])
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            result = np.array(torch.argmax(probs, dim=-1)[0])
            pred_sentiments.append(result)
            result = np.array(list(map(lambda elem: polarity_map_reverse.get(elem), result)))

            # Display results in string
            filtered_tokens = tokens[
                (tokens != '[CLS]')
                & (tokens != '[UNK]')
                & (tokens != '[SEP]')
                & (tokens != '[PAD]')
            ]
            filtered_result = result[
                (tokens != '[CLS]')
                & (tokens != '[UNK]')
                & (tokens != '[SEP]')
                & (tokens != '[PAD]')
            ]

            print('\n', sentence)
            merge_tokens(filtered_tokens, filtered_result)

    pred_sentiments = np.array(pred_sentiments)
    true_sentiments = np.array(true_sentiments)
    true_sentiments, pred_sentiments = post_process(np.array(true_sentiments),
                                                    np.array(pred_sentiments))

    # https://huggingface.co/spaces/evaluate-metric/seqeval
    if lang == 'ko':
        metric = load_metric("seqeval")
        result = metric.compute(predictions=pred_sentiments, references=true_sentiments)
        print(result)


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

    args = parser.parse_args()
    args = Arguments.instance(args)

    # Fine-tuning
    if args.args.train:
        train_aspect_sentimental_classifier()

    # Evaluate
    if args.args.eval and args.args.model_path:
        evaluate_aspect_sentimental_classifier()
