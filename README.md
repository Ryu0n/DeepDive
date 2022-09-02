# ABSA  
## Description
- Aspect-Based Sentimental Analysis for English and Korean  
- English datasets from "SemEval 2016 : Task 5"
- Korean datasets is available at https://corpus.korean.go.kr/request/corpusRegist.do
- You can run this code at `src/model.py`
- You can apply this repository to another task based on `TokenClassification`. (e.g. NER)  

## Usage  
```
python model.py --train=[bool] --eval=[bool] --lang=[str] --model_path=[str]
```
- `--train` : `True` or `False`  
- `--eval` : `True` or `False`  
- `--lang` : `en` or `ko`  
- `--model_path` : checkpoint path

## Future Work
- Wrapping with docker for comfortable
- Add another PLM Classes (e.g. ELECTRA) in `src/utils.py`