import torch
import numpy as np
from typing import List
from fastapi import FastAPI
from dto import NERParams
from transformers import ElectraTokenizerFast, ElectraForTokenClassification
# from torch.cuda import is_available
from torch.backends.mps import is_available


def get_labels_dict():
    labels_dict = dict()
    with open('labels_selectstar.txt', 'r') as f:
        labels = list(map(lambda label: label.replace('\n', ''), f.readlines()))
        for i, label in enumerate(labels):
            labels_dict[label] = i
    return labels_dict


app = FastAPI()
device = 'mps' if is_available() else 'cpu'
labels_dict = get_labels_dict()
labels_dict_inv = {v: k for k, v in labels_dict.items()}
model = ElectraForTokenClassification.from_pretrained("model/ner_ElectraForTokenClassification_epoch_4_avg_loss_0.039.pt")
tokenizer = ElectraTokenizerFast.from_pretrained("beomi/KcELECTRA-base-v2022")
model.to(device)


async def show_merged_sentence(tokenizer: ElectraTokenizerFast, sentence: str, result: np.ndarray):
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
        if len(span_indices) == 0:
            continue
        span_indices = [min(span_indices), max(span_indices)]
        merged_token_dict = {
            'merged_token': merged_token.strip().replace('##', ''),
            'span_indices': span_indices,
            'sentiment': labels_dict_inv.get(sentiment)
        }
        merged_token_dicts.append(merged_token_dict)

    merged_sentence, end_index = '', 0
    for merged_token_dict in merged_token_dicts:
        start, end = merged_token_dict.get('span_indices')
        gap = ' '*(start - end_index)
        if merged_token_dict.get('sentiment') != 'O':
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


async def predict(model, tokenizer, sentence: str):
    inputs = tokenizer.encode_plus(
        sentence,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    inputs.to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    result = torch.argmax(logits, dim=-1)
    logits = logits.detach().cpu()
    result = result.detach().cpu().tolist()[0]
    return result, result

@app.post('/predict')
async def predict_named_entities(params: NERParams):
    response = list()
    sentences = params.sentences
    for sentence in sentences:
        result, result = await predict(model, tokenizer, sentence)
        tag_info = await show_merged_sentence(tokenizer=tokenizer, sentence=sentence, result=result)
        entities = [[span.get('merged_token'), span.get('sentiment'), span.get('span_indices')] for span in tag_info.get('spans') if span.get('sentiment') != 'O']
        not_entities = [[span.get('merged_token'), span.get('sentiment'), span.get('span_indices')] for span in tag_info.get('spans') if span.get('sentiment') == 'O']
        response.append(
            {
                "sentence": sentence,
                "merged_sentence": tag_info.get('merged_sentence'),
                "entities": entities,
                "not_entities": not_entities
            }
        )
    return response
