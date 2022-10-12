import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from preprocess import get_images, labels


def save_checkpoint(result):
    with open('label.json', 'w') as f:
        json_val = json.dumps(result, indent=4)
        f.write(json_val)


def read_checkpoint():
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
    ckpt = read_checkpoint()
    batches = list(zip(*get_images(scaling=False)))

    try:
        for i, (img_path, image) in enumerate(batches):

            if img_path in ckpt.keys():
                print(f'\n{img_path} is already exists!')
                is_change = input("Want to change? [y/n] : ")
                if is_change == 'y':
                    ckpt[img_path] = custom_imshow(image)
                continue

            print(f'\n[{i}/{len(batches)}] {img_path}')
            ckpt[img_path] = custom_imshow(image)

    except KeyboardInterrupt:
        save_checkpoint(ckpt)

    save_checkpoint(ckpt)


def validate_labels(label_num):
    ckpt = read_checkpoint()
    img_paths = [img_path for img_path, label in ckpt.items() if label == str(label_num)]
    for i, img_path in enumerate(img_paths):
        print(f'\n[{i} / {len(img_paths)}]')
        img = np.array(Image.open(img_path))
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    # label_images()
    validate_labels(label_num=4)
