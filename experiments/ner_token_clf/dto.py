from typing import List
from pydantic import BaseModel


class NERParams(BaseModel):
    sentences: List[str]