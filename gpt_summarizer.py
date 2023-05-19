import os
import openai
from typing import List


openai.api_key = os.getenv("OPENAI_API_KEY")


def initialize_prompt(
        documents: List[str],
) -> str:
    description = '아래의 문장들을 요약해주세요.'
    shots_for_prompt = '\n'.join(documents)
    return f'{description}\n{shots_for_prompt}\n요약 내용 : '


def summarize_by_openai(
        model: str,
        prompt: str,
        temperature: float,
        max_token: int
):
    return openai.Completion.create(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_token,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1
    )

async def async_summarize_by_openai(
        model: str,
        prompt: str,
        temperature: float,
        max_token: int
):
    return await openai.Completion.acreate(
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_token,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1
    )