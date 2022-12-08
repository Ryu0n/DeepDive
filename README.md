# Motivation
```
NOTE  
한국어는 교착어 특성상 여러 품사가 띄어쓰기 구분 없이 어절 단위로 합쳐져 있기 때문에 세부적인 분석이 어렵습니다.
반면, 영어는 띄어쓰기 단위로 품사가 구분되어 있기 때문에 ABSA와 같이 TokenClassification이 유리합니다.

ex)
영어 : .. (pancake, '명사') (is, '조사') ..
한국어 : .. 팬케이크는 .. 
```
- 기존의 BERT 모델은 품사 정보를 반영하지 않았습니다.
- 해당 모델은 ```BertTokenizerFast```로 분해된 token들에 ```kiwi```형태소 분석기의 품사를 태깅합니다.
- 모델의 embedding 레이어에 각 토큰들의 품사 임베딩을 반영했습니다.  

# Usage
```
git clone https://github.com/Ryu0nPrivateProject/POSBert.git
```
```python
from transformers import BertTokenizerFast
from kiwipiepy import Kiwi
from custom_bert_tokenizer import tokenize
from custom_bert import CustomBertForTokenClassification


sentences = [
    '나는 사과를 먹었다.',
    '이번 이태원 대참사 진짜 장난 아니네..',
    '너무 힘든데 게임하는건 좋아유'
]
inputs = tokenize(bert_tokenizer=BertTokenizerFast.from_pretrained('klue/bert-base'),
                  kiwi_tokenizer=Kiwi(),
                  sentences=sentences)
model = CustomBertForTokenClassification.from_pretrained('klue/bert-base', num_labels=4)
outputs = model(**inputs)
print(outputs)

"""
TokenClassifierOutput(loss=None, logits=tensor([[[-0.2775, -0.2969, -0.0903, -0.0018],
         [-0.0091,  0.2871, -0.7079,  0.2197],
         [ 0.4855,  0.3676, -0.2845, -0.3408],
         ...,
         [-0.2271, -0.2526, -0.2464,  0.0590],
         [ 0.0728, -0.2573, -0.0605, -0.0643],
         [-0.0497, -0.2890, -0.2162, -0.1672]],

        [[-0.8450,  0.0222, -0.2058,  0.2605],
         [-0.3437, -0.1106, -1.0901, -0.4831],
         [-0.4604, -0.0313, -0.1097,  0.1768],
         ...,
         [-0.8348, -0.4279,  0.1969, -0.4718],
         [-0.9340, -0.3948,  0.2108, -0.5739],
         [-0.6525, -0.4227, -0.0456, -0.0488]],

        [[-1.1676,  0.0696, -0.0715,  0.4199],
         [-0.8934,  0.2599, -1.0831,  0.6113],
         [-0.8213,  0.1867, -0.2176, -0.0370],
         ...,
         [-0.4724, -0.1521,  0.3033,  0.0240],
         [-0.3632, -0.2972,  0.2225, -0.0881],
         [-0.6073, -0.2911,  0.4057,  0.0735]]], grad_fn=<ViewBackward0>), hidden_states=None, attentions=None)
"""
```

# Future Work
- 일부 코드 리팩토링
- ELECTRA 버전 구현