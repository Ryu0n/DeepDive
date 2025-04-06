import numpy as np


class HashFunction:
    def __init__(self, dim: int, seed: int):
        self.dim = dim
        np.random.seed(seed)
        self.vector = np.random.randn(self.dim)

    def __call__(self, vector: np.ndarray) -> int:
        dot_product = np.dot(self.vector, vector)
        return int(dot_product >= 0)


class CompositeHashFunction:
    def __init__(self, hash_functions: list[HashFunction]):
        self.hash_functions = hash_functions

    def __call__(self, vector: np.ndarray) -> tuple:
        return tuple(h(vector) for h in self.hash_functions)


class LSHPreprocessor:
    def __init__(self, L: int, k: int, dim: int):
        self.L = L
        self.k = k
        self.hash_tables = [{} for _ in range(L)]
        self.composite_hash_functions = [
            CompositeHashFunction(
                hash_functions=[HashFunction(dim, seed=i * j + k) for j in range(k)]
            )
            for i in range(L)
        ]

    def preprocess(self, dataset: list[np.ndarray]) -> tuple:
        for vector in dataset:
            for i in range(self.L):
                g_j = self.composite_hash_functions[i]  # k개의 해시 함수 집합
                hash_value = g_j(vector)
                self.hash_tables[i].setdefault(hash_value, []).append(vector)
        return self.hash_tables, self.composite_hash_functions


class LSHRetriever:
    def __init__(
        self, hash_tables: list, composite_hash_functions: list, threshold: float = 1.5
    ):
        self.hash_tables = hash_tables
        self.composite_hash_functions = composite_hash_functions
        self.threshold = threshold
        print(f"Hash tables:\n{hash_tables}")

    def query(self, vector: np.ndarray):
        for i in range(len(self.hash_tables)):
            hash_table = self.hash_tables[i]
            g_j = self.composite_hash_functions[i]
            hash_value = g_j(vector)
            retrieved_vectors = hash_table.get(hash_value, [])
            if not retrieved_vectors:
                print(f"Hash table {i}: No vectors found for hash value {hash_value}")
                continue
            print(
                f"Hash table {i}: Retrieved {len(retrieved_vectors)} vectors for hash value {hash_value}"
            )
            for v in retrieved_vectors:
                distance = np.linalg.norm(vector - v)
                print(
                    f"query vector: {vector}, retrieved vector: {v}, distance: {distance}"
                )
                # L2 distance between query vector and retrieved vector
                if distance <= self.threshold:
                    print(f"Found similar vector: {v} with distance {distance}")


if __name__ == "__main__":
    dim = 5
    L = 2
    k = 3

    query_vector = np.random.randn(dim)

    dataset = [np.random.randn(dim) for _ in range(10)]

    hash_tables, composite_hash_functions = LSHPreprocessor(
        L=L, k=k, dim=dim
    ).preprocess(dataset)

    LSHRetriever(hash_tables, composite_hash_functions).query(query_vector)
