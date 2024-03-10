import base64
import numpy as np
from fastapi import FastAPI
from dto import *
from autoencoder import *


app = FastAPI()


@app.post("/dimensionality_reduce")
def dimension_reduce(sentence_vector_param: SentenceVectors):
    """
    Reduce vector dimension
    :param sentence_vector_param: MUST Convert to bytes from np.ndarray (USE np.ndarray.tobytes())
    :return: MUST Convert to np.ndarray from bytes (USE np.frombuffer(bytes, dtype=np.float32))
    """
    vector_bytes: str = sentence_vector_param.vector_bytes
    decoded_vector_bytes: bytes = base64.b64decode(vector_bytes)
    sentence_vectors: np.ndarray = np.frombuffer(decoded_vector_bytes, dtype=np.float64)
    # This process is essential by process of converting bytes
    sentence_vectors = sentence_vectors.reshape((-1, sentence_vector_param.vector_dim))
    ae = train_auto_encoder(sentence_vectors)
    compressed_sentence_vectors: np.ndarray = compress_vector(ae=ae,
                                                              sentence_vectors=sentence_vectors)
    return base64.b64encode(compressed_sentence_vectors.tobytes()).decode('utf-8')
