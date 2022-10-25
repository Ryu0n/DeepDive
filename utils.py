import os
import json
import zipfile
import shutil
import numpy as np
from PIL import Image
from glob import glob

labels = {
    1: '정상',
    2: '디폴트스팸(성인물/성인용품/부동산/용역/광고대행/종교/철학/도박/대출)',
    3: '무의미'
}


def get_image_paths(from_mongo=False):
    if from_mongo:
        return [p for p in glob('mongo_images/*') if not os.path.isdir(p)]
    return [p for p in glob('instagram/**', recursive=True) if not os.path.isdir(p)]


def get_images(from_mongo=False, rescaled_pixels=500):
    """

    :return: images shape -> (n_samples, width, height, channels)
    """
    img_paths = get_image_paths(from_mongo)
    return img_paths, [np.array(Image.open(img_path)).resize((rescaled_pixels, rescaled_pixels)) for img_path in img_paths]


def get_images_classification_result():
    """
    최종 분류된 결과(spam_images/class_num/*)를 딕셔너리 형태로 반환
    :return:
    """
    content = dict()
    for img_path in glob('spam_images/**', recursive=True):
        if os.path.isdir(img_path):
            continue
        class_num = img_path.split('/')[1]
        content[img_path] = class_num
    return content


def read_label_json():
    try:
        with open('label.json', 'r') as f:
            json_val = ''.join(f.readlines())
            return json.loads(json_val)
    except Exception as e:
        print(e)
        return dict()


def save_label_json(result):
    with open('label.json', 'w') as f:
        json_val = json.dumps(result, indent=4)
        f.write(json_val)


def compress_spam_images():
    shutil.make_archive('spam_images', 'zip', 'spam_images')


def extract_spam_images():
    zip_file = zipfile.ZipFile('spam_images.zip')
    zip_file.extractall('spam_images')
    zip_file.close()

