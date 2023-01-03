import os
import json
import random
import torch.nn as nn
from tqdm import tqdm
from src.utils import Arguments
from src.utils import polarity_map


def _get_json_file(file_name: str):
    cur_dir = os.path.abspath(os.curdir)
    return os.path.join(cur_dir, f'ko_data/{file_name}')


def _load_json_dict(file_name: str):
    with open(_get_json_file(file_name), 'r') as f:
        json_dict = json.load(f)
        return json_dict


def parse_json_dict(tokenizer, file_name: str):
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}  # id : word
    json_dict = _load_json_dict(file_name)
    documents = json_dict.get('document')
    rows = []
    for document in tqdm(documents):
        sentences = document.get('sentence')
        for sentence in sentences:
            sentence_text = sentence.get('sentence_form').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            opinions = sentence.get('opinions')
            tokenized_text = tokenizer.encode_plus(sentence_text, return_offsets_mapping=True, padding='max_length', truncation=True)
            tokenized_text_ids = tokenized_text.get('input_ids')
            tokenized_text_offsets = tokenized_text.get('offset_mapping')
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


def add_additional_data(tokenizer):
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}  # id : word
    rows = []
    file_name = 'pang10.jsonl'
    with open(_get_json_file(file_name), 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line_dict = json.loads(line)
            sentence = line_dict.get('data').replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
            labels = line_dict.get('label')
            tokenized_text = tokenizer.encode_plus(sentence, return_offsets_mapping=True, padding='max_length', truncation=True)
            tokenized_text_ids = tokenized_text.get('input_ids')
            tokenized_text_offsets = tokenized_text.get('offset_mapping')
            tokens = [vocab.get(tokenized_text_id) for tokenized_text_id in tokenized_text_ids]
            sentiments = []

            for i, token in enumerate(tokens):
                sentiment = polarity_map.get('unrelated')
                token_offset = tokenized_text_offsets[i]

                if token not in tokenizer.special_tokens_map.values():
                    for opinion in labels:
                        polarity = opinion[2].lower()
                        start = int(opinion[0])
                        end = int(opinion[1])
                        if start <= token_offset[0] < end:
                            sentiment = polarity_map.get(polarity)
                else:
                    sentiment = nn.CrossEntropyLoss().ignore_index  # -100

                sentiments.append(sentiment)
            rows.append([sentence, sentiments])
        return rows


def down_sampling(rows: list, is_additional: bool = False):
    sampled_rows = []
    rows = [[sentence_text, sentiments] for sentence_text, sentiments in rows if
            1 in sentiments or 2 in sentiments or 3 in sentiments]
    for sentence_text, sentiments in rows:
        total_sentiments = sum([sentiments.count(sent) for sent in [0, 1, 2, 3]])
        num_unrelated = sentiments.count(0)
        unrelated_ratio = num_unrelated / total_sentiments
        if unrelated_ratio > 0.85:
            continue
        sentiment_distribution = [sentiments.count(sent) for sent in [1, 2, 3]]
        num_negative, num_neutral, num_positive = sentiment_distribution
        num_sentiments = sum(sentiment_distribution)
        positive_ratio = num_positive / num_sentiments
        if positive_ratio > 0.5 and is_additional:
            continue
        sampled_rows.append([sentence_text, sentiments])
    return sampled_rows


def read_train_dataset(write=True, num_test_samples=50):
    tokenizer = Arguments.instance().tokenizer
    file_name = 'sample'
    original_rows = parse_json_dict(tokenizer, file_name + '.json')
    train_rows, test_rows = original_rows[num_test_samples:], original_rows[:num_test_samples]
    train_rows = down_sampling(train_rows)

    additional_rows = add_additional_data(tokenizer)
    additional_train_rows, additional_test_rows = additional_rows[num_test_samples:], additional_rows[:num_test_samples]
    additional_train_rows = down_sampling(additional_train_rows, True)

    train_rows.extend(additional_train_rows)
    test_rows.extend(additional_test_rows)

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
