import json
import requests
import numpy as np
import base64


def request_compression(sentence_vectors: np.ndarray):
    """

    :param sentence_vectors: (batch_size, dim) dim=300
    :return:
    """
    byte_a: bytes = sentence_vectors.tobytes()
    url = "http://127.0.0.1:8000"
    query_string = "/dimensionality_reduce"
    data = {
        "vector_bytes": base64.b64encode(byte_a).decode('utf-8'),  # 'AACAPwAAAEAAAEBA'
        "vector_dim": sentence_vectors.shape[-1]
    }
    response = requests.post(url=url + query_string,
                             data=json.dumps(data))
    compressed_vectors = np.frombuffer(base64.b64decode(response.content), dtype=np.float32)
    compressed_vectors = compressed_vectors.reshape((sentence_vectors.shape[0], -1))
    return compressed_vectors


if __name__ == "__main__":
    sentence_vectors = np.random.random((3, 300))
    compressed_vectors = request_compression(sentence_vectors=sentence_vectors)
    print(compressed_vectors.shape)
