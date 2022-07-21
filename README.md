# Custom_NER

- 🤗`Huggingface Tranformers`🤗 라이브러리를 이용하여 구현
- KoBERT, KoELECTRA (+ CRF)를 이용한 한국어 Named Entity Recognition Task
- 기존 KoBERT에 CRF(Conditional Random Field)를 붙인 커스텀 클래스 정의
- monologg님의 KoBERT-NER를 커스터마이징
- FastAPI를 사용하여 모델 로드
- 모두의 말뭉치 (./data_preprocess) Fine-Tuning
- 토큰 단위 태깅 기능 추가 (형태소 분석기 품사 활용)

## Dependencies
- torch==1.4.0
- transformers==2.10.0
- seqeval>=0.0.12
- pytorch-crf==0.7.2
- fastapi==0.68.1
- uvicorn==0.15.0


## How to use This Repository
1. Traning & Evaluation 과정을 통해 학습 모델 파일 생성
2. Prediction 과정을 통해 uvicorn ASGI(Async Server Gateway Interface)과 FastAPI 웹 서버를 구동
3. POST Request

## Training & Evaluation

```bash
# KoBERT, KoELECTRA (for token classification) Only Command
$ CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ./data_preprocess --model_type kobert --do_train --do_eval --write_pred
$ CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ./data_preprocess --model_type koelectra-base-v3  --do_train --do_eval --write_pred

# Custom classes
$ CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ./data_preprocess --model_type kobert-crf --model_dir ./kobert_crf_model --do_train --do_eval --pred_dir ./preds_crf --write_pred

# KoBERT
$ CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ./data_preprocess --model_type kobert-torchcrf --model_dir ./kobert_crf_model --do_train --do_eval --pred_dir ./preds_torchcrf --write_pred
$ CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ./data_preprocess --model_type kobert-lstm-torchcrf --model_dir ./kobert_crf_model --do_train --do_eval --pred_dir ./preds_lstm_torchcrf --write_pred
$ CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ./data_preprocess --model_type kobert-bilstm-torchcrf --model_dir ./kobert_crf_model --do_train --do_eval --pred_dir ./preds_bilstm_torchcrf --write_pred

# KoELECTRA
$ CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ./data_preprocess --model_type koelectra-base-v3-torchcrf --model_dir ./kobert_crf_model --do_train --do_eval --pred_dir ./preds_elec_torchcrf --write_pred
$ CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ./data_preprocess --model_type koelectra-base-v3-lstm-torchcrf --model_dir ./kobert_crf_model --do_train --do_eval --pred_dir ./preds_elec_lstm_torchcrf --write_pred
$ CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ./data_preprocess --model_type koelectra-base-v3-bilstm-torchcrf --model_dir ./kobert_crf_model --do_train --do_eval --pred_dir ./preds_elec_bilstm_torchcrf --write_pred
```

## Prediction
```bash
$ uvicorn app:app --reload
```

```python
# POST REQUEST 
# host_name:port/named_entity_recognition
# Request Body Example
{
  "sentences": [
    "성규는 런던에서 빵을 좋아한다.", 
    "한소희는 성규를 혜화동에서 좋아한다.",
    "나는 서울에서 널 본순간 사랑에 빠졌어",
    "넌 러시아에 있는 집에 가서 잠을 자고 싶다",
    "데이터 분석가 슈퍼마리오는 점프를 6번 했다.",
    "2022년 나는 이동욱을 롤 모델로 삼았지만, 현실은 그렇지 않다."
  ]
}
```

## References
- [monologg/KoBERT-NER](https://github.com/monologg/KoBERT-NER)
- [Implementing a linear-chain Conditional Random Field(CRF) in PyTorch](https://towardsdatascience.com/implementing-a-linear-chain-conditional-random-field-crf-in-pytorch-16b0b9c4b4ea)
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS](https://arxiv.org/pdf/2003.10555.pdf)
