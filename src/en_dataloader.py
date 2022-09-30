import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from transformers import BertTokenizerFast

from src.utils import Arguments
from src.utils import polarity_map


def _get_xml_file(file_name: str):
    cur_dir = os.path.abspath(os.curdir)
    return os.path.join(cur_dir, f'en_data/{file_name}')


def _get_element_tree(file_name: str):
    xml_file = _get_xml_file(file_name)
    tree = ET.parse(xml_file)
    return tree


def parse_element_tree(file_name: str):
    tokenizer_name = Arguments.instance().args.tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)
    vocab = tokenizer.get_vocab()
    vocab = {v: k for k, v in vocab.items()}
    tree = _get_element_tree(file_name)
    root = tree.getroot()
    rows = []

    for review in root:
        sentences = list(review)[0]
        for sentence in sentences:

            sentence = list(sentence)
            if len(sentence) != 2:
                continue
            text, opinions = sentence
            text = text.text

            tokenized_text = tokenizer.encode_plus(text, return_offsets_mapping=True, padding='max_length')
            tokenized_text_ids = tokenized_text.get('input_ids')
            tokenized_text_offsets = tokenized_text.get('offset_mapping')
            tokens = [vocab.get(tokenized_text_id) for tokenized_text_id in tokenized_text_ids]
            sentiments = []

            for i, token in enumerate(tokens):
                sentiment = polarity_map.get('unrelated')
                token_offset = tokenized_text_offsets[i]

                if token not in tokenizer.special_tokens_map.values():
                    for opinion in opinions:
                        opinion: Element
                        polarity = opinion.get('polarity')
                        start = int(opinion.get('from'))
                        end = int(opinion.get('to'))
                        if start <= token_offset[0] < end:
                            sentiment = polarity_map.get(polarity)
                else:
                    sentiment = -100

                sentiments.append(sentiment)
            rows.append([text, sentiments])
    return rows


def read_train_xml(write=True):
    file_name = 'ABSA16_Restaurants_Train_SB1_v2'
    rows = parse_element_tree(f'{file_name}.xml')
    with open(_get_xml_file(f'{file_name}.txt'), 'w') as f:
        for text, sentiments in rows:
            sentiments_str = ' '.join(map(str, sentiments))
            if write:
                f.write(f'{text}\t{sentiments_str}\n')
            yield text, sentiments


def read_test_xml():
    tree = _get_element_tree('EN_REST_SB1_TEST.xml')
    root = tree.getroot()
    for review in root:
        sentences = review[0]
        for sentence in sentences:
            text = sentence[0].text
            yield text


if __name__ == "__main__":
    for text, sentiments in read_train_xml(write=True):
        print(text, sentiments)

