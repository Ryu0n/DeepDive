import os
import json
import glob
import boto3
from tqdm import tqdm
from utils import read_label_json


def secret_information():
    with open('secret_s3.json', 'r') as f:
        secret = json.load(f)
        return secret


def s3_resource(secret):
    s3 = boto3.resource(
        's3',
        aws_access_key_id=secret.get('access_key_id'),
        aws_secret_access_key=secret.get('secret_access_key'),
        region_name=secret.get('region_name')
    )
    return s3


def download_images(s3, secret, limit=100):
    """
    s3 bucket에 존재하는 이미지 다운로드
    일자별로 limit 개의 이미지 다운로드 (label.json에 포함된 이미지 포함)
    :param s3:
    :param secret:
    :param limit: 일자별 이미지 수
   :return:
    """
    # 버킷 인스턴스 생성
    bucket_name = secret.get('bucket_name')
    bucket = s3.Bucket(bucket_name)

    # label.json에 포함된 이미지 우선 다운로드
    label_json = read_label_json()
    for img_path in tqdm(label_json, leave=True, desc='Download images from label.json'):
        dir_name = os.path.dirname(img_path)
        if not dir_name.startswith("instagram"):
            # s3에서 받은 이미지만 다운로드
            continue
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        bucket.download_file(
            Key=img_path,
            Filename=img_path
        )

    # 일자별로 limit 개수만큼 채워지도록 다운로드
    for obj in tqdm(bucket.objects.filter(Prefix='instagram'), leave=True, desc='Download images until meet limitation'):
        key_dir = obj.key
        dir_name = os.path.dirname(key_dir)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if len(glob.glob(dir_name + '/*')) > limit:
            # 일자별로 limit개 이상의 이미지 수집을 하지 않음
            continue
        if not os.path.isdir(key_dir):
            bucket.download_file(
                Key=key_dir,
                Filename=key_dir
            )


if __name__ == "__main__":
    secret = secret_information()
    s3 = s3_resource(secret)
    download_images(s3, secret)
