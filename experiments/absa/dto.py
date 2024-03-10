from typing import List
from pydantic import BaseModel


class TokenSentimentPredictParams(BaseModel):
    sentences: List[str]


class DocumentSentimentPredictParams(BaseModel):
    target_keyword: str
    sentences: List[str]
