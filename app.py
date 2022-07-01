import subprocess
from dto import Sentences
from fastapi import FastAPI


app = FastAPI()


def input_sentences(sentences: Sentences):
    with open('sample_pred_in.txt', 'w') as f:
        f.truncate(0)
        for sentence in sentences.sentences:
            f.write(f'{sentence}\n')


def show_result():
    with open('sample_pred_out.txt', 'r') as f:
        for line in f.readlines():
            yield line


@app.post("/named_entity_recognition")
async def read_root(sentences: Sentences):
    input_sentences(sentences)
    subprocess.call('python predict.py', shell=True)
    result = [line for line in show_result()]
    return result
