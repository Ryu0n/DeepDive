import ast
import time
import json
import requests
from pymongo import MongoClient
from konlpy.tag import Kkma, Okt, Mecab
from transformers import ElectraTokenizer
from pynori.korean_analyzer import KoreanAnalyzer


def collect_samples(num_samples=200):
    """
    MongoDB 로부터 데이터 수집
    """
    data = {
      "sentences": []
    }

    host = 'mongodb://k8s.mysterico.com/'
    port = 31687
    username = 'khan'
    password = 'khanterico'
    client = MongoClient(host=host, port=port, username=username, password=password)
    db = client.get_database('social-monitor-prod')
    comments = db.get_collection('Comment')
    for i, doc in enumerate(comments.find()):
        if i == num_samples:
            break
        content = doc.get('content')
        data.get('sentences').append(content)
    return data


def write_input():
    """
    MongoDB 로부터 수집된 데이터 샘플 텍스트에 입력
    """
    data = collect_samples()
    with open('sample_pred_in.txt', 'w') as f:
        f.truncate(0)
        for i, sentence in enumerate(data.get('sentences')):
            print(i)
            f.write(f'{sentence}\n')


def read_input():
    """
    샘플 텍스트 읽기
    """
    data = {
      "sentences": []
    }
    with open('sample_pred_in.txt', 'r') as f:
        for sentence in f.readlines():
            data.get('sentences').append(sentence)
    return data


def test_tokenize(tokenizer):
    """
    Konlpy 형태소 분석기 샘플 텍스트 테스트
    """
    with open('sample_pred_in.txt', 'r') as f:
        for line in f.readlines():
            print(line)
            print(tokenizer.pos(line), '\n')


def test_tokenizer(tokenizer, sentence):
    """
    Konlpy 형태소 분석기 문장 단위 테스트
    """
    print(tokenizer.pos(sentence))


def test_nori_tokenizer(sentence):
    """
    Nori 형태소 분석기 테스트
    """
    nori = KoreanAnalyzer(
        infl_decompound_mode='NONE',
        discard_punctuation=True
    )
    result = nori.do_analysis(sentence)
    term, tag = map(lambda k: result.get(k), ('termAtt', 'posTagAtt'))
    result_ = list(zip(term, tag))
    print(result_)


def test_ner_request():
    """
    NER Docker container 테스트 (샘플 텍스트)
    """
    data = read_input()
    URL = 'http://0.0.0.0:8000/named_entity_recognition'
    res = requests.post(URL, data=json.dumps(data))
    print(res, res.text)
    res = ast.literal_eval(res.text)
    for t in res:
        print(t)


def test_sns_nori_tokenizer(sentence: str):
    """
    Mysterico API 테스트
    :params sentence: 'okt' or 'nori'
    :params normalize: 'true' or 'false'
    """
    start = time.time()
    data = {"text": sentence}
    URL = 'http://k8s.mysterico.com:31516/async/sns_tokenizer/tokenize/nori'
    res = requests.post(URL, data=json.dumps(data))
    t = time.time() - start
    print(f'time : {t}\n{res}\n{res.text}')


def test_sns_okt_tokenizer(sentence: str, normalize: str = 'false'):
    """
    Mysterico API 테스트
    :params sentence: 'okt' or 'nori'
    :params normalize: 'true' or 'false'
    """
    start = time.time()
    data = {"text": sentence}
    URL = f'http://k8s.mysterico.com:31516/async/sns_tokenizer/tokenize/okt?normalize={normalize}'
    res = requests.post(URL, data=json.dumps(data))
    t = time.time() - start
    print(f'time : {t}\n{res}\n{res.text}')


if __name__ == "__main__":
    # sentence = "cj온마트"
    # sentence = '먹었어요'
    sentence = "이물질 나와서 개빡치는데 개같네요 진짜로 곰표맥주 많다 차박객 2020년 3,000원 삼백오천만묶음 관리사무소 감바스 쉬림프"
    # sentence = "근데 cj온마트인데 cj기프트카드로 결제가 안되는게 흠이네요."
    test_sns_okt_tokenizer(sentence, 'false')
    # test_tokenizer(Okt(), sentence)
    