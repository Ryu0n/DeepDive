# ABSA  
## Description
- Aspect-Based Sentimental Analysis for English and Korean  
- English datasets from "SemEval 2016 : Task 5"
- Korean datasets is available at https://corpus.korean.go.kr/request/corpusRegist.do
- You can run this code at `src/model.py`
- You can apply this repository to another task based on `TokenClassification`. (e.g. NER)  

## Usage  
Run this command in `ABSA` directory.  

```
python src/model.py --train=[bool] --eval=[bool] --lang=[str] --model_path=[str] --tokenizer=[str] --extractor=[bool]
```  

- `--train` : `True` or `False` / REQUIRED  
- `--eval` : `True` or `False` / REQUIRED  
- `--lang` : `en` or `ko` / OPTION  
- `--model_path` : checkpoint path / OPTION (REQUIRED WHEN `--train` is `False` and `--eval` is `True`)  
- `--tokenizer` : tokenizer name in huggingface (correspond to model class) / `bert-base-multilingual-cased` / REQUIRED
- `--extractor` : `True` or `False` / either ATE(Aspect Term Extractor) or Sentimental Classifier (num of classes 2 or 4)

## Future Work
- Wrapping with docker for comfortable
- Add another PLM Classes (e.g. ELECTRA) in `src/utils.py`