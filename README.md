# ABSA  
## Description
- Aspect-Based Sentimental Analysis for English and Korean  
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
