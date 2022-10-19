import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from preprocess import get_image_paths, labels


def save_label_json(result):
    with open('label.json', 'w') as f:
        json_val = json.dumps(result, indent=4)
        f.write(json_val)


def read_label_json():
    try:
        with open('label.json', 'r') as f:
            json_val = ''.join(f.readlines())
            return json.loads(json_val)
    except:
        return dict()


def custom_imshow(image):
    plt.imshow(image)
    plt.show(block=False)
    plt.pause(0.2)
    question = json.dumps(labels, indent=4, ensure_ascii=False) + '\nAnswer : '
    label = input(question)
    plt.close()
    return label


def label_images():
    label_json = read_label_json()
    img_paths = get_image_paths()
    try:
        for i, img_path in enumerate(img_paths):
            if img_path in label_json.keys():
                continue
            image = np.array(Image.open(img_path))
            print(f'\n[{i}/{len(img_paths)}] {img_path}')
            label_json[img_path] = custom_imshow(image)

    except KeyboardInterrupt:
        save_label_json(label_json)

    save_label_json(label_json)


def validate_labels(label_num):
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
    save_label_json(label_json)


if __name__ == "__main__":
    # label_images()
    validate_labels(label_num=2)
