import os
import queue
import multiprocessing as mp
from multiprocessing import Lock
from sentence_transformers import SentenceTransformer


def wait_for_encode(device: str, input_queue: mp.Queue, output_queue: mp.Queue):
    model: SentenceTransformer = load_embedding_model(device)
    mutex = Lock()
    with mutex:
        while True:
            try:
                text_chunk = input_queue.get()
                embeddings = model.encode(
                    sentences=text_chunk,
                    device=device
                )
                output_queue.put(embeddings)
            except queue.Empty:
                break


def spawn_model_process(device: str):
    ctx = mp.get_context('spawn')
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    proc = ctx.Process(
        target=wait_for_encode,
        args=(device, input_queue, output_queue),
        daemon=True
    )
    return proc, input_queue, output_queue


def load_embedding_model(device: str=None) -> SentenceTransformer:
    model_checkpoint = os.environ["BASE_MODEL_CHECKPOINT"]
    if device is not None:
        embedder = SentenceTransformer(model_checkpoint).to(device)
    embedder = SentenceTransformer(model_checkpoint)
    return embedder
