import json
import os.path
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from utils import get_image_paths, read_label_json, save_label_json, labels


def custom_imshow(image):
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(0.2)
    question = json.dumps(labels, indent=4, ensure_ascii=False) + '\nAnswer : '
    label = input(question)
    plt.close()
    return label


def label_images(from_mongo=False):
    """
    이미지 라벨링 후, label.json에 결과 저장
    :param from_mongo: mongoDB로부터 다운로드 받은 이미지 여부 (True : s3, False: mongo)
    :return:
    """
    label_json = read_label_json()
    img_paths = get_image_paths(from_mongo=from_mongo)
    try:
        for i, img_path in enumerate(img_paths):
            if img_path in label_json.keys():
                continue
            print(f'\n[{i}/{len(img_paths)}] {img_path}')
            image = np.array(Image.open(img_path))
            label_json[img_path] = custom_imshow(image)

    except KeyboardInterrupt:
        save_label_json(label_json)
        copy_images_to_spam_images_directory()
    save_label_json(label_json)
    copy_images_to_spam_images_directory()


def validate_labels(label_num):
    """
    label.json의 라벨링 결과 검수
    :param label_num:
    :return:
    """
    label_json = read_label_json()
    img_paths = [img_path for img_path, label in label_json.items() if label == str(label_num)]
    try:
        for i, img_path in enumerate(img_paths):
            print(f'\n[{i} / {len(img_paths)}] {img_path}')
            image = np.array(Image.open(img_path))
            label = custom_imshow(image)
            label_json[img_path] = label
    except KeyboardInterrupt:
        save_label_json(label_json)
        copy_images_to_spam_images_directory()
    save_label_json(label_json)
    copy_images_to_spam_images_directory()


def copy_images_to_spam_images_directory():
    """
    label.json에 있는 이미지들을 최종적으로 spam_images/class_num/ 디렉터리 아래로 복사하여 분류
    :return:
    """
    label_json = read_label_json()
    for img_path, label in tqdm(label_json.items(), leave=True, desc='Copying images'):
        img_base_path = os.path.basename(img_path)
        destination = f'spam_images/{label}/{img_base_path}'
        img_dir_path = os.path.dirname(destination)
        if not os.path.exists(img_dir_path):
            os.makedirs(img_dir_path)
        if not os.path.exists(destination):
            shutil.copyfile(img_path, destination)


if __name__ == "__main__":
    # label_images(from_mongo=False)
    # validate_labels(label_num=2)
    copy_images_to_spam_images_directory()
