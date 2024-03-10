import re
import json
import requests
from typing import List


def tokenize_by_okt(sentence: str) -> str:
    """
    OKT 토크나이저 API 요청
    :param sentence:
    :return:
    """
    url = 'http://127.0.0.1:8080/tokenize'
    data = {
        "sentence": sentence
    }
    response = requests.post(url, data=json.dumps(data))
    return response.content.decode('utf-8')


def string_to_list(string):
    """
    '["나","는","사과","를","먹었다","."]' -> ["나", "는", "사과", "를", "먹었다", "."]
    :param string:
    :return:
    """
    tokens = string[1:-1].split(',')
    return list(map(lambda token: re.sub('"', '', token), tokens))


def filter_stopwords(sentence: str) -> List[str]:
    """
    불용어 제거
    :param sentence:
    :return:
    """
    token_string = tokenize_by_okt(sentence)
    tokens = string_to_list(token_string)
    with open('stopwords.txt', 'r') as f:
        stopwords = list(map(lambda stopword: stopword.replace('\n', ''), f.readlines()))
    sanitized_tokens = list(filter(lambda token: token not in stopwords, tokens))
    return sanitized_tokens


def filter_special_characters(sentence: str) -> str:
    """
    문장 입력받아 특수문자 제거 후 반환
    :param sentence:
    :return:
    """
    return re.sub('[^가-힣0-9a-zA-Z\s]', '', sentence)
