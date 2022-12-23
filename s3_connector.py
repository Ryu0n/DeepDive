import os
import json
import boto3
import subprocess


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


def upload_model_checkpoint(file_name: str):
    """
    모델 가중치 폴더 압축 후 S3 버킷에 업로드
    :param file_name: 모델 가중치 폴더 이름 (.pt)
    :param version:
    :return:
    """
    def compress_model_checkpoint(file_name: str):
        compressed_filename = f'{os.path.basename(file_name)}.tar.gz'
        subprocess.run([
            'tar', '-zcvf', compressed_filename, file_name
        ])
        return compressed_filename

    compressed_filename = compress_model_checkpoint(file_name)
    secret = secret_information()
    s3 = s3_resource(secret=secret)
    bucket = s3.Bucket(secret.get("bucket_name"))
    bucket.upload_file(compressed_filename,
                       f'ABSA/{compressed_filename}')


def download_model_checkpoint(model_checkpoint: str):
    """

    :param model_checkpoint:
    :return:
    """
    def extract_model_checkpoint(compressed_filename: str):
        subprocess.run([
            'tar', '-zxvf', compressed_filename
        ])

    secret = secret_information()
    s3 = s3_resource(secret)
    bucket = s3.Bucket(secret.get("bucket_name"))
    compressed_filename = os.path.basename(model_checkpoint)
    if os.path.exists(compressed_filename):
        return
    bucket.download_file(model_checkpoint, compressed_filename)
    extract_model_checkpoint(compressed_filename)
    return compressed_filename


if __name__ == "__main__":
    upload_model_checkpoint('electra_token_cls_epoch_4_loss_0.18488148148148145.pt')
    # download_model_checkpoint('ABSA/electra_token_cls2.pt_1.0.0.tar.gz')
