from fastapi import FastAPI
from dto import Sentences
from dataloader import dataloader


app = FastAPI()


def input_sentences():
    return dataloader(is_train=True, batch_size=16)


def save_senteces(sentences: Sentences):

    pass


@app.post("/tokenize")
def tokenize_server(sentences: Sentences):
    pass
