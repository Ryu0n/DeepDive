from typing import List
from pydantic import BaseModel


class Sentences(BaseModel):
    sentences: List[str]