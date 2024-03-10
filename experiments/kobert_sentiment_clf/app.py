from fastapi import FastAPI
from dto import Sentences
from transformers import BertForSequenceClassification, BertTokenizer


app = FastAPI()


@app.post('/analysis')
def analysis_semtiment(sentences: Sentences):
    model_checkpoint = "monologg/kobert"
    model = BertForSequenceClassification.from_pretrained(model_checkpoint)
    tokenizer = BertTokenizer.from_pretrained(model_checkpoint)
    sentences = sentences.sentence
    inputs = tokenizer(sentences,
                       padding='max_length',
                       return_tensors='pt')
    outputs = model(**inputs)
    sentiments = outputs.logits.argmax(dim=-1)
    sentiments = sentiments.detach().cpu().tolist()

    response = list()
    for sentence, sentiment in zip(sentences, sentiments):
        result = {
            "sentence": sentence,
            "sentiment": "positive" if sentiment == 1 else "negative"
        }
        response.append(result)

    return response
