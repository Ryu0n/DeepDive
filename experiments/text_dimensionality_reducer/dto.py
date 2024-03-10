from pydantic import BaseModel


class SentenceVectors(BaseModel):
    vector_bytes: str
    vector_dim: int
