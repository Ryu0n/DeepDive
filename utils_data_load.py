import os
import json

from tqdm import tqdm
from glob import glob
from konlpy.tag import Okt
from utils import load_tokenizer


def _proj_dir():
    return '/'.join(os.path.abspath(os.path.curdir).split('/')[:6])


def get_json_paths(d='data_v1'):
    json_dir = os.path.join(_proj_dir(), d)
    return glob(f'{json_dir}/*.json')


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def encoding_to_ner_tag(sentence_dict: dict):
    """
    데이터 가공 기준 : http://air.changwon.ac.kr/?page_id=10
    - 어절 단위 분리
    - 한 어절에서 여러 개체명 태그가 속해있을 경우 먼저 표현된 태그

    '멕시코 파키스탄 방글라데시 베트남 역시 해외파견 근로자의 송금이 한몫을 한다.'
    -> LC LC LC LC O O CV O O O O

    :param d:
    :return:
    """
    result = []
    sentence = sentence_dict.get('form')
    words = sentence.split()

    # if sentence.endswith('.'):
    #     # 구두점 (punctuation) 분리
    #     sentence = sentence[:-1] + ' .'

    ne_dict = {}
    nes: list = sentence_dict.get('NE')
    for ne in nes:
        ne_words = ne.get('form').split()
        ne_label = ne.get('label')
        for i, ne_word in enumerate(ne_words):
            ne_dict[ne_word] = (ne_label, i)  # index for Begin, Inner tags

    for word in words:
        word_label = None
        for ne_word, (ne_label, i) in ne_dict.items():
            if ne_word in word:
                word_label = f'{ne_label}-{"B" if i == 0 else "I"}'
                result.append(word_label)
                ne_dict.pop(ne_word)
                break
        if word_label is None:
            result.append('O')

    return ' '.join(result)


def encoding_json():
    for json_path in tqdm(reversed(get_json_paths('data_v1'))):
        txt_file_name = os.path.basename(json_path).replace('json', 'tsv')
        txt_file_name = os.path.join('data_preprocess', txt_file_name)
        file = open(txt_file_name, 'w')
        json_dict = load_json(json_path)
        documents = json_dict.get('document')
        for document in documents:
            sentences = document.get('sentence')
            for sentence in sentences:
                if sentence.get('form') == '':
                    continue
                result = encoding_to_ner_tag(sentence)
                data = sentence.get('form') + '\t' + result + '\n'
                if data != '\t':
                    file.write(data)
        file.close()


def collect_labels():
    txt_file_name = os.path.join('data_preprocess', 'label.txt')
    file = open(txt_file_name, 'w')

    ne_set = set()
    for json_path in tqdm(reversed(get_json_paths('data_v1'))):
        json_dict = load_json(json_path)
        documents = json_dict.get('document')
        for document in documents:
            sentences = document.get('sentence')
            for sentence in sentences:
                for ne in sentence.get('NE'):
                    ne_set.add(f'{ne.get("label")}-B')  # Begin tag
                    ne_set.add(f'{ne.get("label")}-I')  # Inner tag

    unk_token = 'UNK'
    out_token = 'O'
    data = '\n'.join([unk_token] + [out_token] + sorted(list(ne_set)))
    file.write(data)
    file.close()


collect_labels()
encoding_json()
