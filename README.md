# ABSA  
## Description
- Aspect-Based Sentimental Analysis for Korean  
- Korean datasets is available at https://corpus.korean.go.kr/request/corpusRegist.do
- You can run this code at `src/model.py`
- You can apply this repository to another task based on `TokenClassification`. (e.g. NER)  

## Usage  
Run this command in `ABSA` directory.  

```
python src/model.py --train=[bool] --eval=[bool] --model_path=[str] --tokenizer_name=[str]
```  

- `--train` : `True` or `False` / REQUIRED  
- `--eval` : `True` or `False` / REQUIRED  
- `--model_path` : checkpoint path / OPTION (REQUIRED WHEN `--train` is `False` and `--eval` is `True`)  
- `--tokenizer_name` : tokenizer name in huggingface (correspond to model class) / `bert-base-multilingual-cased` / REQUIRED

## Future Work
- Wrapping with docker for comfortable
- Add another PLM Classes (e.g. ELECTRA) in `src/utils.py`

## Samples  
```python
# 타겟 키워드 : 삼쩜삼

[
  {
    "sentence": "[일반] 삼쩜삼도 사실 별거 아니던데? 그냥 세무사 중개 플랫폼일 뿐임",
    "merged_sentence": "[[일반] 삼쩜삼 : negative]도 사실 별거 아니던데? 그냥 세무사 중개 플랫폼일 뿐임",
    "document_sentiment": "negative"
  },
  {
    "sentence": "[일반] 삼쩜삼이랑 수백개씩 기장공장 하는 세무사직원이랑 다른게 뭐냐? 세무사가 직접 안 하고 직원이 하는건 불법명의 아니냐? 앞으로는 세무사가 직접 안 하면 납세자가 신고하는 시대가 곧 올거다 지금 괜찮다고 나중에도 괜찮다고 생각하면 큰 코 다친다",
    "merged_sentence": "[[일반 : negative]] [삼쩜삼 : negative]이랑 수백개씩 [기장공장 : negative] 하는 [세무사직원이 : negative]랑 다른게 뭐냐? 세무사가 직접 안 하고 [직원이 : negative] 하는건 불법명의 아니냐? 앞으로는 세무사가 직접 안 하면 [납세자 : neutral]가 신고하는 시대가 곧 올거다 지금 괜찮다고 나중에도 괜찮다고 생각하면 큰 코 다친다",
    "document_sentiment": "negative"
  },
  {
    "sentence": "울엄마가 윤석열찍으면 용산 써밋오피스텔 증여해준다고해서 윤석열찍었는데 내일생에서 가장잘한거같어 윤석열 사랑해",
    "merged_sentence": "울엄마가 [윤석열 : positive]찍으면 [용산 써밋오피스텔 : positive] 증여해준다고해서 [윤석열 : positive]찍었는데 내일생에서 가장잘한거같어 [윤석열 : positive] 사랑해",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "검찰 사위없는 사람은 억울해서 살겠냐 공범들은 다 감빵갔는데 윤석열 장모 최은순만 무죄ㅋㅋㅋㅋ",
    "merged_sentence": "[검찰 사위 : negative]없는 사람은 억울해서 살겠냐 공범들은 다 감빵갔는데 [윤석열 장모 최은순 : negative]만 무죄ㅋㅋㅋㅋ",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "개시발 윤석열 장모 무죄확정이네 ㅡㅡ",
    "merged_sentence": "[개시발 윤석열 장모 : negative] 무죄확정이네 ㅡㅡ",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "윤석열 참모 뒤에 숨지 않겠다더니 바이든 김은혜한테 뒤집어씌우고 만5세입학 박순애한테 뒤집어 씌우는 미친인간임",
    "merged_sentence": "[윤석열 : negative] 참모 뒤에 숨지 않겠다더니 바이든 [김은혜 : negative]한테 뒤집어씌우고 [만5세입학 박순애 : negative]한테 뒤집어 씌우는 [미친인간 : negative]임",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "맥도날드가 그렇게 먹고 싶었냐? [자동차 인도주행]",
    "merged_sentence": "[맥도날드 : negative]가 그렇게 먹고 싶었냐? [[자동차 인도주행 : neutral]]",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "한식들 솔직히 존나 뻔하잖아 쌀밥에 밑반찬 찌개 고기 그런걸 누가 먹음 그냥 간단하게 먹지",
    "merged_sentence": "[한식들 : negative] 솔직히 존나 뻔하잖아 [쌀밥에 밑반찬 찌개 고기 : negative] 그런걸 누가 먹음 그냥 간단하게 먹지",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "[일반] 파월 새끼 고용률 딸 드럽게 치네.. 실상은 맥도날드, 버거킹 하루 4시간 알바생들 가지고 무슨 고용률 딸 치면서 고용이 강하다고 하노 ㅋㅋㅋㅋㅋㅋ 진짜 양산의 그새끼같은 짓만 골라 하노...ㅋㅋㅋㅋㅋㅋ",
    "merged_sentence": "[일반] [파월 새끼 고용률 딸 : negative] 드럽게 치네.. 실상은 [맥도날드 : negative], [버거킹 : negative] 하루 4[시간 알바생 : negative]들 가지고 무슨 [고용률 딸 : negative] 치면서 고용이 강하다고 하노 ㅋㅋㅋㅋㅋㅋ 진짜 양산의 [그새끼 : negative]같은 짓만 골라 하노...ㅋㅋㅋㅋㅋㅋ",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "[일반] 윤두창 찢보다 못한 새끼네ㅋㅋㅋㅋ\n전국지표 여론조사\n윤석열 평가\n긍정 34% : 56% 부정\n이재명 평가\n긍정 36% : 51% 부정\n이 새끼 진짜 병신 아니냐ㅋㅋㅋ 찢한테 지네",
    "merged_sentence": "[일반] [윤두창 찢 : negative]보다 못한 새끼네ㅋㅋㅋㅋ [전국지 : neutral]표 [여론조사 윤석열 평가 : neutral] 긍정 34% : 56% 부정 [이재명 : negative] 평가 긍정 36% : 51% 부정 이 새끼 진짜 [병신 : negative] 아니냐ㅋㅋㅋ [찢 : negative]한테 지네",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "윤석열은 입만열면 자동으로 거짓말한다 법과원칙대로 한다면서=주가조작 김건희는 수사도 안하고, 이재명마누라 밥먹는것까지 수사한것 또하고 또하고 수십번한다 돈아겨 쓴다면서=검찰230명증원하고 엄청난 돈들어 이사하고 리모델링하고 이상한부서 새로만들고 레고랜드사태로 채권발행하고 부자는 세금줄어주고",
    "merged_sentence": "[윤석열은 : negative] 입만열면 자동으로 거짓말한다 [법과원칙 : negative]대로 한다면서=[주가조작 김건희는 : negative] 수사도 안하고, [이재명마누라 : negative] 밥먹는것까지 수사한것 또하고 또하고 수십번한다 돈아겨 쓴다면서=[검찰 : negative]230명증원하고 엄청난 돈들어 이사하고 [리모 : negative]델링하고 이상한[부서 : negative] 새로만들고 [레고랜드사태 : negative]로 [채권 : negative]발행하고 [부자는 세금 : negative]줄어주고",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "롯데리아 개씹노맛 왜먹는지 모르겠다 맥도날드는 기생충나오고 버거킹이 선녀다",
    "merged_sentence": "[롯데리아 : negative] 개[씹노맛 : negative] 왜먹는지 모르겠다 [맥도날드 : negative]는 기생충나오고 [버거킹 : positive]이 선녀다",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "맥도날드 너무 맛있어요! 해시브라운도 너무 맛있고",
    "merged_sentence": "[맥도날드 : positive] 너무 맛있어요! [해시브라운 : positive]도 너무 맛있고",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "삼쩜삼 영국 진출 했다는데 그 세금 환급해주는 데... 영국 진출 개쩌네",
    "merged_sentence": "[삼쩜삼 : neutral] 영국 진출 했다는데 그 세금 환급해주는 데... [영국 진출 : positive] 개쩌네",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "와 납세자연맹이랑 삼쩜삼(자비스앤빌런즈) 손잡았네 획기적인 세무비용 절감이 목표. 세무업무하는 세무사 회계사 변호사 어케되냐이제.    ㄷㄷ",
    "merged_sentence": "와 [납세자 : positive][연맹 : neutral]이랑 [삼쩜삼(자비스앤빌런즈) : positive] 손잡았네 획기적인 [세무비용 : negative] 절감이 목표. [세무업무 : negative]하는 [세무사 회계사 변호사 : negative] 어케되냐이제.    ㄷㄷ",
    "document_sentiment": "positive"
  },
  {
    "sentence": "삼쩜삼 광고가 괘씸한 이유 최근 5년 내 소득이 없음 ",
    "merged_sentence": "[삼쩜삼 광고 : negative]가 괘씸한 이유 최근 5년 내 소득이 없음",
    "document_sentiment": "negative"
  },
  {
    "sentence": "정근영의 강아지 치타는 너무 사랑스럽다 ㅎㅎ 근데 우리집 고양이는 정근영처럼 못생겼다.,.",
    "merged_sentence": "[정근영의 강아지 치타는 : positive] 너무 사랑스럽다 ㅎㅎ 근데 우리집 [고양이 : negative]는 [정근영 : negative]처럼 못생겼다.,.",
    "document_sentiment": "neutral"
  },
  {
    "sentence": "겨울이 오면 저는 매년 눈치 싸움을 해요.. 어찌나 쌀쌀한 바람은 재빠르게 알아차리는지 올 해도 여전히 저의 피부는 건조함 한가득이었어요 ㅋㅋ 사실 이젠 놀랍지도 않고 이미 n년간 경험해봤던 것이기 때문에 나름대로 올 해는 미리 준비를 해놨던 것이 다행이라 생각이 드는데요~ 어릴 때 부터 워낙 민감성이라 안써본 제품들이 없기도 했고 또 남들 다 좋다고 소문난 것들을 구입해보면 막상 저한텐 드라마틱하게 잘 맞는게 없더라구요? 진짜 가지가지한다.. 생각을 하면서도 결국엔 내 피부이기 때문에 절대 놓을 수 없어서 이번에도 속는 셈 치고 더하르나이 제품을 처음으로 구입해봤어요! 저는 시카이드 라인에 크림을 구입해 봤는데 조금 더 묵직한 형태로 밤이 있어서 고민을 하다가 데일리로 쓰기엔 크림 형태가 무난하지 않을까? 싶어서 우선은 크림을 결정했던 것 같아요~",
    "merged_sentence": "겨울이 오면 저는 매년 눈치 싸움을 해요.. 어찌나 쌀쌀한 [바람 : neutral]은 재빠르게 알아차리는지 올 해도 여전히 저의 [피부 : negative]는 건조함 한가득이었어요 ㅋㅋ 사실 이젠 놀랍지도 않고 이미 n년간 경험해봤던 것이기 때문에 나름대로 올 해는 미리 준비를 해놨던 것이 다행이라 생각이 드는데요~ 어릴 때 부터 워낙 민감성이라 안써본 제품들이 없기도 했고 또 남들 다 좋다고 [소문난 것들을 : negative] 구입해보면 막상 저한텐 드라마틱하게 잘 맞는게 없더라구요? 진짜 가지가지한다.. 생각을 하면서도 결국엔 내 피부이기 때문에 절대 놓을 수 없어서 이번에도 속는 셈 치고 [더하르나이 제품을 : positive] 처음으로 구입해봤어요! 저는 [시카이드 라인 : positive]에 [크림 : positive]을 구입해 봤는데 조금 더 묵직한 형태로 밤이 있어서 고민을 하다가 데일리로 쓰기엔 [크림 형태 : positive]가 무난하지 않을까? 싶어서 우선은 [크림 : positive]을 결정했던 것 같아요~",
    "document_sentiment": "neutral"
  }
]
```
