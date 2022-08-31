import os
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from transformers import BertTokenizerFast


polarity_map = {
    'positive': 3,
    'neutral': 2,
    'negative': 1,
    'unrelated': 0
}


def _get_xml_file(file_name: str):
    cur_dir = os.path.abspath(os.curdir)
    return os.path.join(cur_dir, f'../data/{file_name}')


def _get_element_tree(file_name: str):
    xml_file = _get_xml_file(file_name)
    tree = ET.parse(xml_file)
    return tree


def parse_element_tree(file_name: str):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')
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

                for opinion in opinions:
                    opinion: Element
                    polarity = opinion.get('polarity')
                    start = int(opinion.get('from'))
                    end = int(opinion.get('to'))
                    if start <= token_offset[0] < end:
                        sentiment = polarity_map.get(polarity)

                sentiments.append(sentiment)
            rows.append([text, sentiments])
    return rows


def read_text(write=False):
    file_name = 'ABSA16_Restaurants_Train_SB1_v2'
    rows = parse_element_tree(f'{file_name}.xml')
    with open(_get_xml_file(f'{file_name}.txt'), 'w') as f:
        for text, sentiments in rows:
            sentiments_str = ' '.join(map(str, sentiments))
            if write:
                f.write(f'{text}\t{sentiments_str}\n')
            yield text, sentiments


if __name__ == "__main__":
    for text, sentiments in read_text(write=True):
        print(text, sentiments)

