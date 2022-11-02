import torch
from transformers import BertForTokenClassification, BertTokenizerFast
from utils import get_labels_dict


labels_dict = get_labels_dict()
labels_dict_inv = {v: k for k, v in labels_dict.items()}


def predict(model, tokenizer, sentence: str):
    inputs = tokenizer.encode_plus(sentence,
                                   return_tensors="pt")
    tokens = inputs.tokens()
    outputs = model(**inputs)
    logits = outputs.logits
    result = torch.argmax(logits, dim=-1).tolist()[0]

    print(f'\n{sentence}')
    for token, label in zip(tokens, result):
        if token in tokenizer.special_tokens_map.values():
            continue
        print(token, labels_dict_inv.get(label))


if __name__ == "__main__":
    model = BertForTokenClassification.from_pretrained("BertForTokenClassification_epoch_4_avg_loss_0.012.pt")
    tokenizer = BertTokenizerFast.from_pretrained("klue/bert-base")

    sentences = [
        "너랑 나랑 서울에서 햄버거 먹은 날 기억해?",
        "맥도날드 진짜 패티에서 기생충 나온거 충격..",
        "이번 이태원 할로윈 축제 진짜 참사",
        "북한이 오늘(2일) 오전 단거리 탄도미사일 3발을 포함한 다종의 미사일을 최소 10여 발 발사한 것으로 파악됐습니다."
    ]

    for sentence in sentences:
        predict(model, tokenizer, sentence)
