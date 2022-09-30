import os
import json
import random
import torch.nn as nn
from transformers import BertTokenizerFast
from src.utils import Arguments
from src.utils import polarity_map


def _get_json_file(file_name: str):
    cur_dir = os.path.abspath(os.curdir)
    return os.path.join(cur_dir, f'ko_data/{file_name}')


def _load_json_dict(file_name: str):
    with open(_get_json_file(file_name), 'r') as f:
        json_dict = json.load(f)
        return json_dict


def parse_json_dict(file_name: str):
    tokenizer_class = Arguments.instance().tokenizer_class
    tokenizer_name = Arguments.instance().args.tokenizer
    tokenizer = tokenizer_class.from_pretrained(tokenizer_name)
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}  # id : word
    json_dict = _load_json_dict(file_name)
    documents = json_dict.get('document')
    rows = []
    for document in documents:
        sentences = document.get('sentence')
        for sentence in sentences:
            sentence_text = sentence.get('sentence_form')
            opinions = sentence.get('opinions')
            tokenized_text = tokenizer.encode_plus(sentence_text, return_offsets_mapping=True, padding='max_length')
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


def train_test_split(rows: list, train_ratio: float):
    train_size = int(len(rows) * train_ratio)
    train_rows, test_rows = rows[:train_size], rows[train_size:]
    return train_rows, test_rows


def down_sampling(rows: list, ratio=1):
    """
    Deal with imbalance problem.
    :param rows:
    :param ratio: multiple of smaller label count
    :return:
    """
    negative, neutral, positive = map(lambda polarity: polarity_map.get(polarity), ['negative', 'neutral', 'positive'])
    negative_rows = [(sentence_text, sentiments) for sentence_text, sentiments in rows if negative in sentiments]
    positive_rows = [(sentence_text, sentiments) for sentence_text, sentiments in rows if positive in sentiments]
    neutral_rows = [(sentence_text, sentiments) for sentence_text, sentiments in rows if neutral in sentiments]

    num_negative, num_positive = int(len(negative_rows) * ratio), len(positive_rows)
    if num_positive < num_negative:
        negative_rows, positive_rows = positive_rows, negative_rows

    down_sampled_rows = dict()
    for _ in range(num_negative):
        if len(positive_rows):
            random_idx = random.randint(0, len(positive_rows)-1)
            random_positive = positive_rows.pop(random_idx)
            down_sampled_rows[random_positive[0]] = random_positive
        if len(neutral_rows):
            random_idx = random.randint(0, len(neutral_rows)-1)
            random_neutral = neutral_rows.pop(random_idx)
            down_sampled_rows[random_neutral[0]] = random_neutral

    for negative_row in negative_rows:
        down_sampled_rows[negative_row[0]] = negative_row

    down_sampled_rows = list(down_sampled_rows.values())
    random.shuffle(down_sampled_rows)
    return down_sampled_rows


def down_sampling_v2(rows: list):
    rows = [[sentence_text, sentiments] for sentence_text, sentiments in rows if 1 in sentiments or 2 in sentiments or 3 in sentiments]
    return rows


def read_train_dataset(write=True, train_ratio=0.8):
    file_name = 'sample2'
    rows = parse_json_dict(file_name+'.json')
    rows = down_sampling_v2(rows)
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
    with open(_get_json_file('sample2_test.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            sentence_text, sentiments = line.split('\t')
            sentiments = sentiments.split(' ')
            sentiments = list(map(int, sentiments))
            yield sentence_text, sentiments


if __name__ == "__main__":
    for sentence_text, sentiments in read_train_dataset(write=True):
        print(sentence_text, sentiments)
