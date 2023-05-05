import os
import math
import numpy as np
from fastapi import FastAPI
from typing import List
from dto.dto import *
from embed.embedder import *


app = FastAPI()
num_worker_container = int(os.environ["NUM_WORKER_CONTAINER"])
process_pool = [
                    spawn_model_process(f'cuda:{i}') 
                    for i in range(num_worker_container)
                ]
for proc, _, _ in process_pool:
    proc.start()


async def split_into_chunks(texts: List[str]) -> List[List[str]]:
    if len(texts) == 0:
        return []
    unit = math.ceil(len(texts) / num_worker_container)
    text_chunks = []
    for i in range(0, len(texts), unit):
        text_chunk = texts[i:i+unit]
        text_chunks.append(text_chunk)
    return text_chunks


@app.post('/get_embedding_mp')
async def request_embedding_mp(params: TextEmbeddingRequestParams):
    embeddings = []
    text_chunks = await split_into_chunks(params.text)    
    for i, text_chunk in enumerate(text_chunks):
        _, input_queue, output_queue = process_pool[i]
        input_queue.put(text_chunk)
    for _, _, output_queue in process_pool:
        while output_queue.empty():
            continue
        while not output_queue.empty():
            embedding = output_queue.get()
            embeddings.append(embedding)
    embeddings = np.concatenate(embeddings, axis=0)
    result = []
    for text, embedding in zip(params.text, embeddings):
        result.append(
            TextEmbedding(
                text=text,
                embedding=embedding.tolist()
            )
        )
    return TextEmbeddingResponse(result=result)
    