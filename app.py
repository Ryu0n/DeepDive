import json
from dto import *
from fastapi import FastAPI
import numpy as np


app = FastAPI()


@app.post("/")
def dimension_reduction(sentence_vectors: SentenceVectors):
    return json.dumps(np.ndarray(sentence_vectors.sentence_vectors))


if __name__ == '__main__':
    print(dimension_reduction())

