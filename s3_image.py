import os
import json
import glob
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


def read_label_json():
    try:
        with open('label.json', 'r') as f:
            json_val = ''.join(f.readlines())
            return json.loads(json_val)
    except Exception as e:
        print(e)
        return dict()


def download_images(s3, secret, limit=1000):
    img_path = './images'
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    # 버킷 인스턴스 생성
    bucket_name = secret.get('bucket_name')
    bucket = s3.Bucket(bucket_name)

    for obj in tqdm(bucket.objects.filter(Prefix='instagram'), leave=True):
        key_dir = obj.key
        file_name = img_path + '/' + key_dir
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if len(glob.glob(dir_name + '/*')) > limit:
            # 일자별로 limit개 이상의 이미지 수집을 하지 않음
            continue
        if not os.path.isdir(file_name):
            bucket.download_file(
                Key=key_dir,
                Filename=file_name
            )


def download_images_in_labeling(s3, secret):
    # label.json
    label_json = read_label_json()
    labeled_img_paths = ['/'.join(labeled_img_path.split('/')[1:]) for labeled_img_path in label_json.keys()]

    img_path = './images'
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    bucket_name = secret.get('bucket_name')
    bucket = s3.Bucket(bucket_name)

    for labeled_img_path in tqdm(labeled_img_paths):
        file_name = img_path + '/' + labeled_img_path
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if os.path.exists(file_name):
            continue
        bucket.download_file(
            Key=labeled_img_path,
            Filename=file_name
        )


def download_images_without_labeling(s3, secret, limit=100):
    label_json = read_label_json()
    labeled_img_paths = ['/'.join(labeled_img_path.split('/')[1:]) for labeled_img_path in label_json.keys()]

    img_path = './images'
    if not os.path.exists(img_path):
        os.mkdir(img_path)

    bucket_name = secret.get('bucket_name')
    bucket = s3.Bucket(bucket_name)
    downloaded_dict = dict()

    for obj in tqdm(bucket.objects.filter(Prefix='instagram'), leave=True):
        key_dir = obj.key
        file_name = img_path + '/' + key_dir
        dir_name = os.path.dirname(file_name)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if len(downloaded_dict.setdefault(dir_name, list())) > limit:
            # 일자별로 다운로드할 이미지 수 제한
            continue
        if key_dir in labeled_img_paths:
            # except_labeling 옵션을 준 경우, 라벨링되지 않은 이미지만 수집
            continue
        if not os.path.isdir(file_name):
            bucket.download_file(
                Key=key_dir,
                Filename=file_name
            )
            # 일자별로 다운로드한 이미지 기록
            downloaded_dict.setdefault(dir_name, list()).append(key_dir)


if __name__ == "__main__":
    secret = secret_information()
    s3 = s3_resource(secret)
    download_images(s3, secret)

    # download_images_in_labeling(s3, secret)
