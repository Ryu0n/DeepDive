import os
import json
import random
import torch.nn as nn
from utils import Arguments
from utils import polarity_map
from tqdm import tqdm


def _get_json_file(file_name: str):
    cur_dir = os.path.abspath(os.curdir)
    return os.path.join(cur_dir, f'ko_data/{file_name}')


def _load_json_dict(file_name: str):
    with open(_get_json_file(file_name), 'r') as f:
        json_dict = json.load(f)
        return json_dict


def parse_json_dict(file_name: str, tokenizer, tokenize_func):
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}  # id : word -> word : id
    json_dict = _load_json_dict(file_name)
    documents = json_dict.get('document')
    rows = []
    for document in tqdm(documents, leave=True):
        sentences = document.get('sentence')
        for sentence in sentences:
            sentence_text = sentence.get('sentence_form')
            opinions = sentence.get('opinions')
            # tokenized_text = tokenizer.encode_plus(sentence_text, return_offsets_mapping=True, padding='max_length', truncation=True)
            tokenized_text = tokenize_func([sentence_text])
            tokenized_text_ids = tokenized_text.get('input_ids')[0]
            tokenized_text_offsets = tokenized_text.get('offset_mapping')[0]
            tokens = [vocab.get(tokenized_text_id) for tokenized_text_id in tokenized_text_ids]
            sentiments = []

            for i, token in enumerate(tokens):
                sentiment = polarity_map.get('unrelated')
                token_offset = tokenized_text_offsets[i]

                if token not in tokenizer.special_tokens_map.values():
                    for opinion in opinions:
                        polarity = opinion.get('polarity')
                        start = int(opinion.get('begin'))
                        end = int(opinion.get('end'))
                        if start <= token_offset[0] < end:
                            sentiment = polarity_map.get(polarity)
                else:
                    sentiment = nn.CrossEntropyLoss().ignore_index  # -100

                sentiments.append(sentiment)
            rows.append([sentence_text, sentiments])
    return rows


def add_additional_data(tokenizer, tokenize_func):
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}  # id : word
    rows = []
    with open(_get_json_file('ABSA_negative_token_labeling_data.jsonl'), 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_dict = json.loads(line)
            sentence = line_dict.get('data').replace('\n', ' ')
            labels = line_dict.get('label')
            # tokenized_text = tokenizer.encode_plus(sentence, return_offsets_mapping=True, padding='max_length', truncation=True)
            tokenized_text = tokenize_func([sentence])
            tokenized_text_ids = tokenized_text.get('input_ids')[0]
            tokenized_text_offsets = tokenized_text.get('offset_mapping')[0]
            tokens = [vocab.get(tokenized_text_id) for tokenized_text_id in tokenized_text_ids]
            sentiments = []

            for i, token in enumerate(tokens):
                sentiment = polarity_map.get('unrelated')
                token_offset = tokenized_text_offsets[i]

                if token not in tokenizer.special_tokens_map.values():
                    for opinion in labels:
                        polarity = opinion[2]
                        start = int(opinion[0])
                        end = int(opinion[1])
                        if start <= token_offset[0] < end:
                            sentiment = polarity_map.get(polarity)
                else:
                    sentiment = nn.CrossEntropyLoss().ignore_index  # -100

                sentiments.append(sentiment)
            rows.append([sentence, sentiments])
        return rows


def down_sampling(rows: list):
    sampled_rows = []
    rows = [[sentence_text, sentiments] for sentence_text, sentiments in rows if 1 in sentiments or 2 in sentiments or 3 in sentiments]
    for sentence_text, sentiments in rows:
        total_sentiments = sum([sentiments.count(sent) for sent in [0, 1, 2, 3]])
        num_unrelated = sentiments.count(0)
        unrelated_ratio = num_unrelated / total_sentiments
        if unrelated_ratio > 0.8 and 1 not in sentiments and 2 not in sentiments:
            continue
        sampled_rows.append([sentence_text, sentiments])
    return sampled_rows


def train_test_split(rows: list, train_ratio: float):
    train_size = int(len(rows) * train_ratio)
    train_rows, test_rows = rows[:train_size], rows[train_size:]
    return train_rows, test_rows


def read_train_dataset(write=True, train_ratio=0.8):
    tokenizer = Arguments.instance().tokenizer
    tokenize_func = Arguments.instance().tokenize_func
    file_name = 'sample'
    rows = parse_json_dict(file_name+'.json', tokenizer, tokenize_func)
    additional_rows = add_additional_data(tokenizer, tokenize_func)
    rows.extend(additional_rows)
    rows = down_sampling(rows)
    random.shuffle(rows)
    train_rows, test_rows = train_test_split(rows, train_ratio)

    # save test text
    if write:
        with open(_get_json_file(file_name+'_test'+'.txt'), 'w') as f:
            for sentence_text, sentiments in test_rows:
                sentiments_str = ' '.join(map(str, sentiments))
                f.write(f'{sentence_text}\t{sentiments_str}\n')

    # save train text
    with open(_get_json_file(file_name+'_train'+'.txt'), 'w') as f:
        for sentence_text, sentiments in train_rows:
            sentiments_str = ' '.join(map(str, sentiments))
            if write:
                f.write(f'{sentence_text}\t{sentiments_str}\n')
            yield sentence_text, sentiments


def read_test_dataset():
    with open(_get_json_file('sample_test.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sentence_text, sentiments = line.split('\t')
            sentiments = sentiments.split(' ')
            sentiments = list(map(int, sentiments))
            yield sentence_text, sentiments


