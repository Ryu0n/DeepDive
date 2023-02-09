import torch
import numpy as np
from fastapi import FastAPI
from app_utils import *
from dto import TokenSentimentPredictParams, DocumentSentimentPredictParams
from transformers import ElectraTokenizerFast, ElectraForTokenClassification


app = FastAPI()
polarity_map = {
    0: 'unrelated',
    1: 'negative',
    2: 'neutral',
    3: 'positive',
}
model = ElectraForTokenClassification.from_pretrained('electra_token_cls_epoch_4_loss_0.23.pt')
tokenizer = ElectraTokenizerFast.from_pretrained('beomi/KcELECTRA-base-v2022')
model.eval()


@app.post('/predict_token_sentiment')
async def predict_token_sentiment(params: TokenSentimentPredictParams):
    """
    각 문장들을 토큰 단위의 감정 분류 결과를 리턴
    :param params:
    :return:
    """
    tag_informs = list()
    for i, sentence in enumerate(params.sentences):
        with torch.no_grad():
            inputs = tokenizer.encode_plus(sentence, return_tensors='pt', padding='max_length', truncation=True)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            result = np.array(torch.argmax(probs, dim=-1)[0])
            result = np.array(list(map(lambda elem: polarity_map.get(elem), result)))
            tag_informs.append(show_merged_sentence(tokenizer, sentence, result))
    return tag_informs


@app.post('/predict_document_sentiment')
async def predict_document_sentiment(params: DocumentSentimentPredictParams):
    """
    각 문장들을 토큰 단위로 감정 분류를 한 뒤,
    타겟 키워드를 중심으로 해당 문장에 대한 감정 분류 결과를 리턴
    :param params:
    :return:
    """
    tag_informs = list()
    for i, sentence in enumerate(params.sentences):
        with torch.no_grad():
            inputs = tokenizer.encode_plus(sentence, return_tensors='pt', padding='max_length', truncation=True)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            result = np.array(torch.argmax(probs, dim=-1)[0])
            result = np.array(list(map(lambda elem: polarity_map.get(elem), result)))
            tag_informs.append(show_merged_sentence(tokenizer, sentence, result))
    document_sentiments = await calculate_sentimental_score(target_keyword=params.target_keyword,
                                                            tag_informs=tag_informs)
    sentiment_results = list()
    for tag_info, sentiment in zip(tag_informs, document_sentiments):
        sentiment_results.append(
            {
                "sentence": tag_info.get('origin_sentence'),
                "merged_sentence": tag_info.get('merged_sentence'),
                "document_sentiment": sentiment
            }
        )
    return sentiment_results
