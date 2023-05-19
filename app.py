from dto import *
from gpt_summarizer import *
from redis_utils import *
from token_counter import *
from fastapi import FastAPI


app = FastAPI()


@app.post('/summarize_text', response_model=TextSummarizeResponse)
async def summarize_text(params: TextSummarizeRequestParams):
    texts: List[str] = params.text if isinstance(params.text, list) else [params.text]
    texts = list(sorted(texts))
    prompt: str = initialize_prompt(texts)
    crc: int = extract_crc_from_string(prompt)

    if is_exist_redis_key(crc):
        return get_result_from_redis(crc)
    
    summarized = summarize_by_openai(
        model=params.model,
        prompt=prompt,
        temperature=params.temperature,
        max_token=calculate_max_token(
            model=params.model, 
            prompt=prompt
        )
    )
    return set_result_to_redis(
        redis_key=crc,
        document=texts,
        prompt=prompt,
        summarized=summarized
    )
