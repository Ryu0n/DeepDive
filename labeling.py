import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from preprocess import get_images, labels


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
    plt.pause(0.5)
    question = json.dumps(labels, indent=4, ensure_ascii=False)
    label = input(question)
    plt.close()
    return label


def label_images():
    label_json = read_label_json()
    batches = list(zip(*get_images(scaling=False)))

    try:
        for i, (img_path, image) in enumerate(batches):

            if img_path in label_json.keys():
                print(f'\n{img_path} is already exists!')
                is_change = input("Want to change? [y/n] : ")
                if is_change == 'y':
                    label_json[img_path] = custom_imshow(image)
                continue

            print(f'\n[{i}/{len(batches)}] {img_path}')
            label_json[img_path] = custom_imshow(image)

    except KeyboardInterrupt:
        save_label_json(label_json)

    save_label_json(label_json)


def validate_labels(label_num):
    label_json = read_label_json()
    img_paths = [img_path for img_path, label in label_json.items() if label == str(label_num)]
    for i, img_path in enumerate(img_paths):
        print(f'\n[{i} / {len(img_paths)}]')
        img = np.array(Image.open(img_path))
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    label_images()
    validate_labels(label_num=4)
