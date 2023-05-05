import os
from dto import *
from fastapi import FastAPI
from text_sanitizer import *
from torch.cuda import is_available
from model import *
from typing import List
from inference_postprocess import *


app = FastAPI()
sanitizer = PretrainTextPreprocessing()
device = 'cuda' if is_available() else 'cpu'
model = load_model(
    model_weight=os.environ["MODEL_WEIGHT"],
    device=device
)


# @app.post("/category_clf", response_model=CategoryClassifyResponse)
# async def classify_category(params: CategoryClassifyRequestParams):
#     texts: List[str] = list(map(sanitizer, params.text))
#     results: List[int] = classify(model=model, texts=texts)
#     return CategoryClassifyResponse(
#         category=os.environ["MODEL_CATEGORY"],
#         result=results
#     )
    

@app.post("/category_clf_chunk", response_model=CategoryClassifyResponse)
async def classify_category_chunk(params: CategoryClassifyRequestParams):
    sanitized_docs: List[str] = list(map(sanitizer, params.text))
    results: List[int] = [await predict_with_postprocess(model=model, document=document) for document in sanitized_docs]
    return CategoryClassifyResponse(
        category=os.environ["MODEL_CATEGORY"],
        result=results
    )


# @app.post("/category_clf_sentence", response_model=CategoryClassifyResponse)
# async def classify_category_chunk(params: CategoryClassifyRequestParams):
#     sanitized_docs: List[str] = list(map(sanitizer, params.text))
#     results: List[int] = [await predict_with_sentences(model=model, document=document) for document in sanitized_docs]
#     return CategoryClassifyResponse(
#         category=os.environ["MODEL_CATEGORY"],
#         result=results
#     )