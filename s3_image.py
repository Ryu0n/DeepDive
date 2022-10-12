import os
import json
from tqdm import tqdm
import boto3


def secret_information():
    with open('secret.json', 'r') as f:
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


def download_images(s3, secret):
    img_path = './images'
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    bucket_name = secret.get('bucket_name')
    bucket = s3.Bucket(bucket_name)
    for obj in tqdm(bucket.objects.filter(Prefix='instagram'), leave=True):
        key_dir = obj.key
        file_name = img_path + '/' + key_dir
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if not os.path.isdir(file_name):
            bucket.download_file(
                Key=key_dir,
                Filename=file_name
            )


if __name__ == "__main__":
    secret = secret_information()
    s3 = s3_resource(secret)
    download_images(s3, secret)
