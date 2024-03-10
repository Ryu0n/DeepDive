from typing import List
from pydantic import BaseModel


class TextEmbeddingRequestParams(BaseModel):
    text: List[str]
    

class TextEmbedding(BaseModel):
    text: str
    embedding: list


class TextEmbeddingResponse(BaseModel):
    result: List[TextEmbedding]
