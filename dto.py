from pydantic import BaseModel, Field
from typing import List
import numpy as np


class SentenceVectors(BaseModel):
    sentence_vectors: np.ndarray = Field(default_factory=lambda: np.zeros(1000))

    class Config:
        arbitrary_types_allowed = True
