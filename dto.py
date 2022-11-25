from typing import List
from pydantic import BaseModel


class Sentences(BaseModel):
    sentence: List[str]
    