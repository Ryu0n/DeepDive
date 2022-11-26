from fastapi import FastAPI
from dto import Sentences
from transformers import BertForSequenceClassification, BertTokenizer


app = FastAPI()


@app.post('/analysis')
def analysis_semtiment(sentences: Sentences):
    load_model = BertForSequenceClassification.from_pretrained('klue/bert-base')
    tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
    tokenized_inputs = tokenizer.encode_plus(sentences,
                                             max_length='max_length',
                                             return_tensors='pt')

    pass
