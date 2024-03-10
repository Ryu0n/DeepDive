import os
import zlib
import json
import redis
from typing import Union, List
from dto import RedisScheme


def redis_connection() -> redis.StrictRedis:
    return redis.StrictRedis(
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        db=os.environ["REDIS_DB_NUM"]
    )


conn: redis.StrictRedis = redis_connection()


def extract_crc_from_string(string: str) -> int:
    bytes_string = bytes(string, encoding='utf8')
    return zlib.crc32(bytes_string)


def set_result_to_redis(
        redis_key: int,
        document: List[str],
        prompt: str,
        summarized: dict = None
):
    value = RedisScheme(
        document=document,
        prompt=prompt,
        summarized=summarized
    ).__dict__
    conn.set(
        name=redis_key,
        value=json.dumps(value)
    )
    conn.expire(
        name=redis_key,
        time=os.environ["REDIS_EXPIRED_SECOND"]
    )
    return value


def get_result_from_redis(redis_key: int) -> dict:
    result = conn.get(name=redis_key)
    return RedisScheme().__dict__ if result is None else json.loads(result)


def is_exist_redis_key(redis_key: Union[str, int]) -> int:
    return conn.exists(str(redis_key))
