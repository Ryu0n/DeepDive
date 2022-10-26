import os
import json
import boto3
from utils import compress_spam_images, extract_spam_images


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


def upload_dataset(s3):
    zip_file_name = 'spam_images.zip'
    if os.path.exists(zip_file_name):
        os.remove(zip_file_name)
    compress_spam_images()
    bucket = s3.Bucket('image-spam-datasets')
    bucket.upload_file(
        Filename=zip_file_name,
        Key=zip_file_name
    )


def download_dataset(s3):
    zip_file_name = 'spam_images.zip'
    bucket = s3.Bucket('image-spam-datasets')
    bucket.download_file(
        Key=zip_file_name,
        Filename=zip_file_name
    )
    extract_spam_images()


if __name__ == "__main__":
    secret = secret_information()
    s3 = s3_resource(secret)
    upload_dataset(s3)
