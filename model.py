import os
import torch
from setfit import SetFitModel
from typing import List


def load_model(model_weight: str, device: str) -> SetFitModel:
    return SetFitModel.from_pretrained(model_weight).to(device)


def classify(model: SetFitModel, texts: List[str]) -> List[int]:
    probs = model.predict_proba(texts)
    indices = probs.argmax(axis=1)
    threshold = float(os.environ["MODEL_THRESHOLD"])
    threshold_indices = torch.tensor(
        [
            indices[i] if prob[indices[i]] > threshold else abs(indices[i]-1) 
            for i, prob in enumerate(probs)
        ]
    )
    return threshold_indices.tolist()
