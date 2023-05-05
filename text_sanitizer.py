import re
import html_text
from unicodedata import normalize as unicode_normalize
from soynlp.normalizer import *
import requests
from kiwipiepy import Kiwi


ses = requests.Session()


class NoneResultError(Exception):
    def __str__(self):
        return '[EMPTY RESULT]'


def float_round_and_parse_str(float_data):
    result = round(float_data, 4)
    return float(result)


def float_list_round_and_parse_str(float_data_list):
    result_list = []
    for tmp_data in float_data_list:
        result_list.append(float(round(tmp_data, 4)))
    return result_list

def number_to_zero(in_str):
    # Fix : TypeError Expected string or bytes-like object
    result = re.sub('\d', '0', str(in_str))
    return result

class PlainTextPreprocessing:

    def __call__(self, text):
        result_str = text
        if not result_str:
            return None
        try:
            result_str = self.extract_plain_text_from_html(result_str)
            # 초성 중성 종성 쪼개진거 합치기
            result_str = unicode_normalize('NFC', result_str)

            #html 태그 제거하다가 특수문자 처리 안된경우
            result_str = self.replace_html_special_char(result_str)
            # delete printed hexa code


            result_str = self.delete_printed_hexa_str(result_str)

            # 한글, 영문, 일부 특수기호 및 감정표현 이모티콘 제외 삭제
            result_str = self.delete_emotion_special_char_duplicate(result_str)
            # print(result_str)
            result_str = self.extract_kor_eng_num_emoji(result_str)
            result_str = self.delete_duplicate_linebreak(result_str)
            result_str = self.delete_duplicate_space(result_str)
            result_str = self.delete_emotion_special_char_duplicate(result_str)
            result_str = result_str.lower()
            result_str = self.delete_fixed_pattern_sns_platform_str(result_str)
            result_str = self.delete_email(result_str)
            result_str = self.delete_url(result_str)
            result_str = self.delete_id_annotation(result_str)

        # 결과가 None이 되는게 아닌 오류 발생시 일단 전처리 PASS 위해
        except Exception as e:
            return None

        return result_str

    @staticmethod
    def delete_printed_hexa_str(in_str):
        result = re.sub(r'\\x[\w]{2}', ' ', in_str)
        return result

    @staticmethod
    def delete_id_annotation(in_str):
        """
        twitter, insta, naver등 id 언급 제거
        @뒤에 숫자, 영문, 한글 연속된 경우 삭제
        """
        result = re.sub(r'@[0-9a-zA-Zㄱ-힣_]{1,51}', '', in_str).strip()

        if not result:
            raise NoneResultError
        return result

    @staticmethod
    def delete_emotion_special_char_duplicate(in_str):

        # result = emoticon_normalize(in_str, num_repeats=2)

        result = repeat_normalize(in_str, num_repeats=2)
        # 한 글자가 3번 이상 연속되는 경우 > 2개로 줄임

        for tmp_repeat in set(re.findall(r'(\S)\1{2,}', result)):
            if not tmp_repeat:
                continue

            regex = r'([' + tmp_repeat + ']){2,}'
            try:
                p = re.compile(regex)
            except re.error as e:
                # 정규표현식용 특수문자 예외처리
                tmp_repeat = f'\\{tmp_repeat}'
                regex = r'([' + tmp_repeat + ']){2,}'
                p = re.compile(regex)

            result = p.sub(r'\1\1', result)
        # print(result)
        # 1~4 글자가 동일하게 반복되는 경우 1개로 삭제
        regex = r'(\S)([\S\W])?([\S\W])?([\S\W])?(\1{1}\2?\3?\4?){3,}'
        p = re.compile(regex)

        result = p.sub(r'\1\2\3\4', result)

        if not result:
            raise NoneResultError

        return result

    @staticmethod
    def extract_kor_eng_num_emoji(in_str):
        """
            남기기
            - \u0020-\u007E : 기본 ASCII 범위 영문자, 문장부호, 숫자 포함
            - \u2010-\u2064 : 추가 수식
            - \u00a9 : copyright
            - \u2600-\u26FF : 전화기 표시, 우산 등 빈출 이모티콘
            - ㄱ-ㅣ가-힣
            - \u1F300-\u1F64F 감정표현 및
            - \u1F900-\u1F9FF 추가 이모티콘

            +++비슷한 감정 축약
            ~~~ U+1F487 까지만 쓰자
        """
        mod_str = re.sub(
            r'[^\u0020-\u007Eㄱ-ㅣ가-힣\u2010-\u2064\u00a9\u2600-\u26FF\U0001F300-\U0001F64F\U0001F900-\U0001F9FF]',
            ' ', in_str).strip()

        # 웃는표정들 하나로 축약
        mod_str = re.sub(
            r'[\U0001F600-\U0001F606\U0001F619-\U0001F61D\U0001F609-\U0001F60E\U0001F638-\U0001F63D\U0001F642\U0001F645-\U0001F647\U0001F64B-\U0001F64C]',
            '\U0001F600', mod_str).strip()

        if not mod_str:
            raise NoneResultError

        return mod_str

    @staticmethod
    def extract_plain_text_from_html(in_str):
        result = html_text.extract_text(in_str)
        if not result:
            raise NoneResultError
        return result

    @staticmethod
    def delete_email(in_str):
        return re.sub(r'[a-zA-Z0-9-\.]+@([a-zA-Z0-9-]+.)+[a-zA-Z0-9-]{2,4}', '메일', in_str)

    @staticmethod
    def delete_url(in_str):
        def pass_exception_case(in_case):
            # 소수점에 대해서 처리X ex) 30.1

            try:

                if not in_case[0]:
                    return in_case[0]

                if in_case[0].find('http') > -1 and in_case[0].find('//') > -1:
                    # print(f'{in_case[0]} > 링크 pass')
                    return ''

                # 짧은 문자열에 대한 처리
                # 1. 짧은 도메인 규칙 + 영문만으로 된 도메인 룰이면서 짧은거 링크로 간주
                if re.fullmatch(r'(([a-zA-Z\-]+\.)+[a-zA-Z\-]{2,5})', in_case[0]):
                    fullmatched_text = re.fullmatch(r'(([a-zA-Z\-]+\.)+[a-zA-Z\-]{2,5})', in_case[0])
                    if len(fullmatched_text[0]) < 7:
                        return ''
                # 도메인 룰에 어긋나지만 너무 짧은건 패스
                elif len(in_case[0]) < 7:
                    # print(f'{in_case[0]} > 짧아 pass')
                    return in_case[0]

                # 정규표현식으로 찾은 표현에 숫자가 글자보다 많은 경우 숫자 소수점 표현으로 간주
                if len(re.sub(r'[^\d\-\~\%\.]', '', in_case[0])) / len(in_case[0]) >= 0.5:
                    # print(f'{in_case[0]} > 숫자 pass')
                    return in_case[0]
                # print(f'{in_case[0]} > 링크')
            except Exception as e:
                print(e)
                print(in_case)

            return ''

        regex = re.compile(
            r'(http[s]?:\/\/((www\.)|(m\.))?|ftp:\/\/(www\.)?|www\.)?([0-9a-zA-Z\%\_\-]+\.)+[0-9a-zA-Z\%\_\-]{2,5}(\/[0-9a-zA-Z\%\_\-]+)*(\.[0-9a-zA-Z\%\_\-]+)?\/?(([?#](([\w%]+)?=[\w\%\+\:\-\.]*[&]?(%26)?)+)|[\?]|([\/a-zA-Z0-9_\-]?)*)*')

        return regex.sub(pass_exception_case, in_str)

    @staticmethod
    def delete_duplicate_linebreak(in_str):
        try:
            in_str = in_str.replace('\\n', ' ')
            if in_str:
                in_str = in_str.replace('\\r', ' ')
        except:
            result = in_str

        result = re.sub(r'[\n\r]{1,}', ' ', in_str)

        if not result:
            raise NoneResultError

        return result

    @staticmethod
    def delete_duplicate_space(in_str):
        result = re.sub(r'[\s\u200b\u200c\ufeff]{2,}', ' ', in_str)
        if not result:
            raise NoneResultError

        return result

    @staticmethod
    def delete_fixed_pattern_sns_platform_str(in_str):
        """
        플랫폼마다 반복되는 문자 제거
        """

        """
        반복 문자 제거
        """
        fixed_str_list = ['dc official app', '[eng]', '[sub]', '비디오를 보시려면 원문링크를 클릭하여 확인하세요.']

        in_str = in_str.lower()
        for fixed_str in fixed_str_list:
            in_str = in_str.replace(fixed_str, ' ')
            if not in_str:
                raise NoneResultError

        """
        반복 패턴 제거
        """
        pattern_list = [
            {
                # 네이버 뉴스
                'pattern': r'[\.\n\r][\S\s]{,5}기자\s메일[\S\s]*무단\s전재\s및\s재배포\s금지',
                'replace': r'.'
            },
            {
                # 네이버 뉴스
                'pattern': r'\.*[\sa-zA-Z0-9ㄱ-힣]*기자[\S\s]*무단\s전재\-재배포\s금지',
                'replace': r'.'
            },
            {
                # 네이버 카페 컨텐츠
                'pattern': r'\[{1,3}content\-element\-\d*\]{1,3}',
                'replace': r' '
            }
        ]
        for tmp_pattern in pattern_list:
            sub_tmp_pattern = re.compile(tmp_pattern['pattern'])
            in_str = sub_tmp_pattern.sub(tmp_pattern['replace'], in_str)

        return in_str

    @staticmethod
    def replace_html_special_char(in_str):
        case_list = [
            {
                'html': '&gt;',
                'text': '>'
            },
            {
                'html': '&lt;',
                'text': '<'
            },
            {
                'html': '&amp;',
                'text': '&'
            },
            {
                'html': '&quot;',
                'text': '"'
            },
            {
                'html': '&#039;',
                'text': "'"
            },
            {
                'html': '&nbsp;',
                'text': " "
            },
            {
                'html': '”',
                'text': '"'
            },
            {
                'html': '’',
                'text': "'"
            }
        ]
        result = in_str
        for tmp_case in case_list:
            result = result.replace(tmp_case['html'], tmp_case['text'])

        return result


class PretrainTextPreprocessing(PlainTextPreprocessing):

    def __call__(self, text):

        result = super().__call__(text)
        try:
            result = self.delete_bracelet(result)
            result = self.delete_hashtag(result)
            result = self.delete_link_changed_text(result)
            result = self.delete_duplicate_space(result).strip()
            result = self.delete_noise_start_doc(result)
            result = self.delete_spare_comma(result)
            result = self.delete_spare_url(result)
            result = self.delete_duplicate_space(result).strip()
            result = self.pass_text(result).strip()
        except NoneResultError:
            return None
            # 결과가 None이 되는게 아닌 오류 발생시 일단 전처리 PASS 위해
        except Exception as e:
            raise TypeError

        return result

    @staticmethod
    def delete_spare_comma(in_str):
        result = re.sub(r'\.( \.)+', '. ', in_str)
        if not result:
            raise NoneResultError
        return result

    @staticmethod
    def delete_spare_url(in_str):
        result = re.sub(r'(\&?[a-zA-Z0-9_%]+=[a-zA-Z0-9_%]+)', '', in_str)
        if not result:
            raise NoneResultError
        return result

    @staticmethod
    def delete_hashtag(in_str):
        result = re.sub(r'\#[A-Za-z0-9ㄱ-ㅣ가-힣#_]+', '', in_str)
        if not result:
            raise NoneResultError
        return result

    @staticmethod
    def delete_noise_start_doc(in_str):
        result = in_str
        while True:
            if len(result) > 2 and result[1] == ' ':
                if not re.findall(r'[a-zA-Z0-9ㄱ-ㅣ가-힣]', result[0]):
                    result = result[2:]
                    continue
            break
        if not result:
            raise NoneResultError
        return result

    @staticmethod
    def delete_link_changed_text(in_str):
        if in_str[-2:] == '링크':
            in_str = in_str[:-2]
        if in_str[:2] == '링크':
            in_str = in_str[2:]
        result = re.sub(r'\s링크\s', ' ', in_str)
        if not result:
            raise NoneResultError
        return result

    @staticmethod
    def delete_bracelet(in_str):

        result = re.sub(r'\[[^\[\]]*\]|\([^()]*\)|\{[^\{\}]*\}|\<[^\<\>]*\>', '', in_str)
        if not result:
            raise NoneResultError

        return result

    @staticmethod
    def delete_fixed_text(in_str):

        fixed_str_list = []
        result = in_str
        for tmp_str in fixed_str_list:
            result = result.replace(tmp_str, ' ')

        return result

    @staticmethod
    def pass_text(in_str):
        # 해당 문자열이 포함된 글은 None값 return
        pass_text_list = ['쿠팡 파트너스의 일환으로 일정액의 수수료를 지급 받을 수 있음', '중고나라 운영정책 :', '댓글은 미션완료로 인정되지 않습니다', ]
        for tmp_pass_text in pass_text_list:
            if in_str.find(tmp_pass_text) > -1:
                return None

        return in_str


def analyze_sentence(in_text):
    if not in_text:
        return None
    kiwi = Kiwi()
    try:
        for _ in range(10):
            in_text = unicode_normalize('NFC',in_text)
            in_text = in_text.replace('\u200b', '').replace('\"','').replace('\'','')
            sentence_list = [sent.text for sent in kiwi.split_into_sents(in_text)]
            break
    except Exception as e:
        print(e)
        return None

    return sentence_list


def chunk_size_sentence_split(in_sentence_list, min_len=60, max_len=200):
    """

    """
    # 텍스트 입력 시, 일정 길이 수준(상황에 따라 변동)으로 문장 나눔

    # 줄나눔
    kss_sentence_splitted_list = in_sentence_list

    sentence_splitted_list = []
    # 너무 긴 길이로 나누어진 문장을 '. ' 단위로 나눔(kss 성능 문제)
    for tmp_splitted_sentence in kss_sentence_splitted_list:
        if len(tmp_splitted_sentence) > max_len:
            sentence_splitted_list.extend([f.strip() + '.' for f in tmp_splitted_sentence.split('. ') if len(f) > 5])
        else:
            sentence_splitted_list.append(tmp_splitted_sentence)

    total_extracted_sentence = []

    tmp_extracted_sentence = ''
    for idx, tmp_splitted_sentence in enumerate(sentence_splitted_list):
        #         print(tmp_splitted_sentence)
        # 한 문장
        if len(sentence_splitted_list) == 1:
            total_extracted_sentence.append(tmp_splitted_sentence.strip())
            break
        # 마지막 문장
        if (idx + 1) == len(sentence_splitted_list):
            #  마지막 문장이 너무 긴경우
            if len(tmp_splitted_sentence) > max_len:
                # 쌓인 문장이 적절한 경우
                if min_len <= len(tmp_extracted_sentence) <= max_len:
                    total_extracted_sentence.append(tmp_extracted_sentence.strip())
                    total_extracted_sentence.append(tmp_splitted_sentence.strip())
            else:
                # 쌓인 문장 + 새 문장 길이가 적절한경우
                if min_len <= len(tmp_extracted_sentence) + len(tmp_splitted_sentence) <= max_len:
                    total_extracted_sentence.append((tmp_extracted_sentence + ' ' + tmp_splitted_sentence).strip())
                # 쌓인 문장 + 새 문장 길이가 적절하지 않은 경우
                else:
                    if len(tmp_extracted_sentence) <= 10 or len(tmp_splitted_sentence) <= 10:
                        total_extracted_sentence.append((tmp_extracted_sentence + ' ' + tmp_splitted_sentence).strip())
                    else:
                        total_extracted_sentence.append(tmp_extracted_sentence.strip())
                        total_extracted_sentence.append(tmp_splitted_sentence.strip())
        # 마지막 문장 아닌 경우
        else:
            # 쌓인 문장 + 새 문장 길이가 적절한 경우

            if min_len <= len(tmp_extracted_sentence) + len(tmp_splitted_sentence) <= max_len:
                tmp_extracted_sentence = tmp_extracted_sentence + ' ' + tmp_splitted_sentence
                total_extracted_sentence.append(tmp_extracted_sentence.strip())
                tmp_extracted_sentence = ''
                continue
            # 쌓인 문장 + 새 문장 길이가 적절하지 않은 경우
            else:
                # 쌓인 문장이 기준보다 큰 경우
                if len(tmp_extracted_sentence) >= max_len:
                    total_extracted_sentence.append(tmp_extracted_sentence.strip())
                    tmp_extracted_sentence = ''
                    # 새로운 문장이 기준보다 큰 경우
                if len(tmp_splitted_sentence) > max_len:
                    total_extracted_sentence.append(tmp_splitted_sentence.strip())
                    tmp_extracted_sentence = ''
                else:
                    # 쌓인 문장이 없을 경우 새로운 문장을 쌓음.
                    if not tmp_extracted_sentence:
                        tmp_extracted_sentence = tmp_splitted_sentence
                        # 쌓인 문장이 있을 경우 새로운 문장 추가
                    else:
                        tmp_extracted_sentence = tmp_extracted_sentence + ' ' + tmp_splitted_sentence
    return total_extracted_sentence
