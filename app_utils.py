import numpy as np
from collections import Counter
from transformers import ElectraTokenizerFast


def show_merged_sentence(tokenizer: ElectraTokenizerFast, sentence: str, result: np.ndarray):
    inputs = tokenizer.encode_plus(sentence,
                                   return_tensors='pt',
                                   padding='max_length',
                                   truncation=True,
                                   return_offsets_mapping=True)
    tokens, offsets = inputs.tokens(), inputs.get('offset_mapping').tolist()[0]
    chunks, chunk, curr_sentiment = list(), list(), result[1]  # 0 is [CLS]
    for token, sentiment, offset in zip(tokens, result, offsets):
        if token in tokenizer.special_tokens_map.values():
            continue
        if curr_sentiment != sentiment:
            chunk.append(curr_sentiment)
            chunks.append(chunk.copy())
            chunk = list()
        chunk.append((token, offset))
        curr_sentiment = sentiment
    if len(chunk) > 0:
        chunk.append(curr_sentiment)
        chunks.append(chunk.copy())

    end_index, merged_token_dicts = 0, list()
    for chunk in chunks:
        merged_token = ''
        span_indices = list()
        tokens, sentiment = chunk[:-1], chunk[-1]
        for token, offset in tokens:
            start, end = offset
            gap = ' '*(start - end_index)
            merged_token += f'{gap}{token}'
            span_indices.extend(offset)
            end_index = end
        span_indices = [min(span_indices), max(span_indices)]
        merged_token_dict = {
            'merged_token': merged_token.strip().replace('##', ''),
            'span_indices': span_indices,
            'sentiment': sentiment
        }
        merged_token_dicts.append(merged_token_dict)

    merged_sentence, end_index = '', 0
    for merged_token_dict in merged_token_dicts:
        start, end = merged_token_dict.get('span_indices')
        gap = ' '*(start - end_index)
        if merged_token_dict.get('sentiment') != 'unrelated':
            merged_token = f'[{merged_token_dict.get("merged_token")} : {merged_token_dict.get("sentiment")}]'
        else:
            merged_token = merged_token_dict.get("merged_token")
        merged_sentence += f'{gap}{merged_token}'
        end_index = end
    tag_info = {
        "origin_sentence": sentence,
        "merged_sentence": merged_sentence,
        "spans": merged_token_dicts
    }
    return tag_info


async def calculate_sentimental_score(target_keyword: str, tag_informs: list):
    document_sentiments = list()
    for tag_info in tag_informs:
        sentiments = list()
        spans = tag_info.get('spans')
        for span in spans:
            merged_token = span.get('merged_token')
            if target_keyword in merged_token:
                sentiment = span.get('sentiment')
                sentiments.append(sentiment)

        counter = Counter(sentiments)
        counter = dict(counter)
        sentimental_score = (counter.setdefault('negative', 0) * -1.0) + (counter.setdefault('positive', 0) * 1.0)
        document_sentiment = 'neutral'
        if sentimental_score > 0:
            document_sentiment = 'positive'
        elif sentimental_score == 0:
            document_sentiment = 'neutral'
        elif sentimental_score < 0:
            document_sentiment = 'negative'
        document_sentiments.append(document_sentiment)

    return document_sentiments
