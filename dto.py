from typing import List
from pydantic import BaseModel


class CategoryClassifyRequestParams(BaseModel):
    text: List[str]


class CategoryClassifyResponse(BaseModel):
    category: str
    result: List[int]
