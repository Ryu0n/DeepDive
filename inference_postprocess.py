import os
import aiohttp
import numpy as np
from typing import List
from itertools import chain
from unicodedata import normalize as unicode_normalize
from setfit import SetFitModel


special_characters = [
    '\u200b',
    '\"',
    '\''
]


def eliminate_special_characters(text: str) -> str:
    """
    특수 문자 제거
    """
    text = unicode_normalize('NFC', text)
    for special_character in special_characters:
        text = text.replace(special_character, '')
    return text


async def split_into_sentences(text: str) -> List[str]:
    """
    문장 단위 분리
    """
    async def request_split_sentence(text: str):
        async with aiohttp.ClientSession() as sess:
            response = await sess.post(
                url='http://k8s.mysterico.com:31516/sns_tokenizer/sentence',
                json={
                    "text": text
                }
            )
            result = await response.json()
            return result.get("sentence")
    return await request_split_sentence(text)
    

def split_long_sentences_into_short(sentence: str, max_len=200) -> List[str]:
    """
    Kiwi 토크나이저로 잘린 문장이 너무 긴 경우에 '. ' 단위로 split
    """
    if len(sentence) <= max_len:
        return [sentence]
    splitted_sentences = sentence.split('. ')  # 문장 단위로 못잘린 경우 구두점으로 나눔
    splitted_sentences = [sentence.strip() for sentence in splitted_sentences]  # 공백 제거
    splitted_sentences = [sentence if sentence.endswith('.') else sentence+'.' for sentence in splitted_sentences]
    return splitted_sentences


def split_into_chunks(sentences: List[str], min_len=60, max_len=200) -> List[str]:
    chunks = []
    tmp_chunk = ''
    while len(sentences) > 0:
        tmp_chunk += f' {sentences.pop(0)}'
        tmp_chunk = tmp_chunk.strip()
        
        # 다음 루프가 존재하지 않을 경우 chunks에 추가 후 종료
        if len(sentences) == 0:
            chunks.append(tmp_chunk)
            tmp_chunk = ''
            break
        
        # chunk가 최소 길이(min_len)을 만족하지 못하면 계속 채워넣음
        if len(tmp_chunk) < min_len:
            # 다음 루프가 존재하지 않으면 chunks에 추가하고 종료
            if len(sentences) == 0:
                chunks.append(tmp_chunk)
                tmp_chunk = ''
            continue
        
        # chunk가 최대 길이(max_len)을 넘으면 chunks에 추가
        if len(tmp_chunk) > max_len:
            chunks.append(tmp_chunk)
            tmp_chunk = ''
            continue
    return chunks
            

async def postprocess(document: str):
    document = eliminate_special_characters(document)
    sentences = await split_into_sentences(document)
    splitted_sentences_list = list(
        chain(
            *[
                split_long_sentences_into_short(sentence) for sentence in sentences
            ]
        )
    )
    chunks = split_into_chunks(splitted_sentences_list)
    return chunks


async def predict_with_postprocess(model: SetFitModel, document: str):
    if document is None:
        return 0
    chunks = await postprocess(document)
    if len(chunks) == 0:
        return 0
    chunks_probs = model.predict_proba(chunks, as_numpy=True)  # (N, num_labels)
    threshold = float(os.environ["MODEL_THRESHOLD"])
    chunks_max_indices = np.argmax(chunks_probs, axis=1)  # (N,)
    threshold_indices = [
        chunks_max_indices[N] if chunk_probs[chunks_max_indices[N]] > threshold else abs(chunks_max_indices[N]-1) 
        for N, chunk_probs in enumerate(chunks_probs)
    ]
    return 1 if 1 in threshold_indices else 0


async def predict_with_sentences(model: SetFitModel, document: str):
    if document is None:
        return 0
    document = eliminate_special_characters(document)
    sentences = await split_into_sentences(document)
    if len(sentences) == 0:
        return 0
    chunks_probs = model.predict_proba(sentences, as_numpy=True)  # (N, num_labels)
    threshold = float(os.environ["MODEL_THRESHOLD"])
    chunks_max_indices = np.argmax(chunks_probs, axis=1)  # (N,)
    threshold_indices = [
        chunks_max_indices[N] if chunk_probs[chunks_max_indices[N]] > threshold else abs(chunks_max_indices[N]-1) 
        for N, chunk_probs in enumerate(chunks_probs)
    ]
    return 1 if 1 in threshold_indices else 0