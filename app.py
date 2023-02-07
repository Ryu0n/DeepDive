import torch
from typing import List
from fastapi import FastAPI
from dto import NERParams
from transformers import ElectraTokenizerFast, ElectraForTokenClassification


def get_labels_dict():
    labels_dict = dict()
    with open('labels_selectstar.txt', 'r') as f:
        labels = list(map(lambda label: label.replace('\n', ''), f.readlines()))
        for i, label in enumerate(labels):
            labels_dict[label] = i
    return labels_dict


app = FastAPI()
labels_dict = get_labels_dict()
labels_dict_inv = {v: k for k, v in labels_dict.items()}


def merge_tokens(tokenizer, tokens: List[str], labels: List[str]):
    result = list()
    for token, label in zip(tokens, labels):
        if token in tokenizer.special_tokens_map.values():
            continue
        if not token.startswith("##"):
            result.append([(token, label)])
            continue
        result[-1].append((token, label))
    entities = list()
    for sub_tokens in result:
        first_sub_token = sub_tokens[0]
        if first_sub_token[1] == 0:
            continue
        filtered_sub_tokens = list()
        for sub_token in sub_tokens:
            token, label = sub_token
            if label == 0:
                break
            filtered_sub_tokens.append((token, labels_dict_inv.get(label)))
        token = ''.join([token.replace('##', '') for token, label in filtered_sub_tokens])
        label = filtered_sub_tokens[0][1]
        entities.append((token, label))
    return entities


def predict(model, tokenizer, sentence: str):
    inputs = tokenizer.encode_plus(sentence,
                                   return_tensors="pt")
    tokens = inputs.tokens()
    outputs = model(**inputs)
    logits = outputs.logits
    result = torch.argmax(logits, dim=-1).tolist()[0]
    entities = merge_tokens(tokenizer, tokens, result)
    return entities


@app.post('/predict')
def predict_named_entities(params: NERParams):
    model = ElectraForTokenClassification.from_pretrained("model/ner_ElectraForTokenClassification_epoch_4_avg_loss_0.039.pt")
    tokenizer = ElectraTokenizerFast.from_pretrained("beomi/KcELECTRA-base-v2022")
    sentences = params.sentences
    result = [
        {
            "sentence": sentence,
            "entities": predict(model, tokenizer, sentence)
        }
        for sentence in sentences
    ]
    return result


if __name__ == "__main__":
    sentences = [
        "너랑 나랑 서울에서 햄버거 먹은 날 기억해?",
        "맥도날드 진짜 패티에서 기생충 나온거 충격..",
        "이번 이태원 할로윈 축제 진짜 참사",
        "북한이 오늘(2일) 오전 단거리 탄도미사일 3발을 포함한 다종의 미사일을 최소 10여 발 발사한 것으로 파악됐습니다.",
        "이태원 참사로 가장 많은 인명피해를 입은 건 20대·30대 젊은층이었습니다.",
        "3년 만에 '거리두기' 없이 맞이하게 된 핼러윈, 들뜬 마음으로 이태원을 찾았던 건 대부분 20대였습니다.",
        '''
        Guo Wengui는 2017년부터 Der Spiegel 및 Voice of America와의 인터뷰를 진행하면서 BBC와도 소통했습니다. 이후에 Guo Wengui는 자신의 Twitter 및 youtubi 계정을 등록 및 개설했으며 "Guo Media" 및 "Ant God" 라이브를 개설했습니다. 센세이션 등의 과대 광고 방식은 해외 온라인 매체를 통해 불법적으로 탄압된 사실을 적극적으로 홍보하고 자랑합니다. 그의 연기 하나하나가 "장난"처럼 보이지만 일부 "바보 친구들"의 식욕을 충족시켜줍니다. 일부 사람들의 눈에는 무가치한 '도망자'에서 '인터넷 유명인사'로 급부상한 Guo Wengui가 정말 인상적입니다.
        '''
    ]
    model = ElectraForTokenClassification.from_pretrained("model/ner_ElectraForTokenClassification_epoch_4_avg_loss_0.039.pt")
    tokenizer = ElectraTokenizerFast.from_pretrained("beomi/KcELECTRA-base-v2022")
    result = [
        {
            "sentence": sentence,
            "entities": predict(model, tokenizer, sentence)
        }
        for sentence in sentences
    ]
    print(result)
