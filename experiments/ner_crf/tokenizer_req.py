import re
import json
import requests
from konlpy.tag import Okt, Mecab
from pynori.korean_analyzer import KoreanAnalyzer


okt_filter_pos = (
    'Josa',
    'Punctuation',
    'Verb',
    'KoreanParticle',
    'Unknown',
    'Stopwords'
)

mecab_select_pos = (
    'NNBC',  # 단위를 나타내는 명사
    'NNG',  # 일반 명사
    'NNP',  # 고유 명사
    'NR',  # 수사
    'SN',  # 숫자
    'MM'  # 관형사
)


nori_select_pos = (
    'NNBC',  # 단위를 나타내는 명사
    'NNG',  # 일반 명사
    'NNP',  # 고유 명사
    'NR',  # 수사
    'SN',  # 숫자
    'MM'  # 관형사
)


def local_okt(word):
    okt = Okt()
    pos = okt.pos(word)
    particles = [p[0] for p in pos if p[1] in okt_filter_pos]
    return pos, okt_filter_pos, particles


def local_mecab(word):
    mecab = Mecab()
    pos = mecab.pos(word)
    particles = [p[0] for p in pos if p[1] not in mecab_select_pos]
    return pos, mecab_select_pos, particles


def local_nori(word):
    nori = KoreanAnalyzer(
        infl_decompound_mode='NONE',  # 굴절어 서브워드 출력 X,
        discard_punctuation=True  # 구두점 제거
    )
    pos = nori.do_analysis(word)
    term, tag = map(lambda k: pos.get(k), ('termAtt', 'posTagAtt'))
    pos = list(zip(term, tag))
    particles = [p[0] for p in pos if p[1] not in nori_select_pos]
    return pos, nori_select_pos, particles


def findall(p, s):
    '''Yields all the positions of
    the pattern p in the string s.'''
    i = s.find(p)
    while i != -1:
        yield i
        i = s.find(p, i+1)


def _revert_stopwords(word, pos):
    word_ = word.lower()
    pos_set = set()
    for token, p in pos:
        token_ = token.lower()
        for i in findall(token_, word_):
            elem = (i, i + len(token_), token_, p)
            pos_set.add(elem)

    pos_set = sorted(pos_set, key=lambda k: k[0])
    # 첫 번째 토큰이 첫 인덱스부터 포함되지 않은 경우
    if len(pos_set):
        first_tag = pos_set[0]
        start = first_tag[0]
        if start != 0:
            elem = (0, start, word[0:start], 'Stopwords')
            pos_set.append(elem)
    # 중간에 누락된 단어 추가 (Stopwords)
    for i in range(1, len(pos_set)):
        curr = pos_set[i]
        prev = pos_set[i-1]
        prev_end = prev[1]
        curr_start = curr[0]
        if curr_start != prev_end:
            elem = (prev_end, curr_start, word[prev_end:curr_start], 'Stopwords')
            pos_set.append(elem)
    # 인덱스 순으로 오름차순 정렬
    pos_set = sorted(pos_set, key=lambda k: k[0])
    # 공백 토큰 무시
    pos_set = list(filter(lambda elem : elem[2] != ' ', pos_set))
    # 인덱스 제거 및 토큰 앞뒤 공백 제거
    pos_set = [(elem[2].strip(), elem[3]) for elem in pos_set]
    particles = [elem[0] for elem in pos_set if elem[1] in okt_filter_pos]
    return pos_set, particles


def request_okt(word, normalize='false'):
    response = _req_okt_tokenizer(word, normalize)
    response = json.loads(response)
    pos = response.get('okt')
    pos = [(p.get('token'), p.get('tag')) for p in pos]
    particles = [p[0] for p in pos if p[1] in okt_filter_pos]
    # print('\npos : ', pos)
    # print('particles : ', particles)
    pos, particles = _revert_stopwords(word, pos)
    # print('\npos : ', pos)
    # print('particles : ', particles)
    return pos, okt_filter_pos, particles


def request_nori(word):
    response = _req_nori_tokenizer(word)
    response = json.loads(response)
    pos = response.get('nori')
    pos = [(p.get('token'), re.sub("\(.*\)", '', p.get('tag'))) for p in pos]
    pos = [(p[0], re.sub(".*/", '', p[1])) for p in pos]
    particles = [p[0] for p in pos if p[1] in nori_select_pos]
    print('\n', pos)
    print(particles)
    return pos, nori_select_pos, particles


def _req_okt_tokenizer(sentence, normalize):
    URL = f'http://k8s.mysterico.com:31516/async/sns_tokenizer/tokenize/okt?normalize={normalize}'
    data = {
        'text': sentence
    }
    response = requests.post(URL, data=json.dumps(data))
    return response.text


def _req_nori_tokenizer(sentence):
    URL = 'http://k8s.mysterico.com:31516/async/sns_tokenizer/tokenize/nori'
    data = {
        'text': sentence
    }
    response = requests.post(URL, data=json.dumps(data))
    return response.text


if __name__ == "__main__":
    # sentence = 'cj온마트'
    # pos = [('cj', 'Noun'), ('마트', 'Verb')]
    # _revert_stopwords(sentence, pos)

    sentence = "근데 cj온마트인데 cj기프트카드로 결제가 안되는게 흠이네요."
    # sentence = "약간의 화요일 보너스 좋다요...."
    # sentence = "이물질 나와서 개빡치는데 개같네요 진짜로 곰표맥주 많다 차박객 2020년 3,000원 삼백오천만묶음 관리사무소 감바스 쉬림프"
    sentence = "입짧은 저희부부 3일저녁에치를 한끼에 다 드심ㅋㅋ 대단"
    sentence = " 한끼에"
    pos, select_pos, particles = request_okt(sentence)
    # print(sentence)
