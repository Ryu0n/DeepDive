import os
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


labels = {
    1: '정상',
    2: '광고성문구',
    3: '부적절한이미지(성인물/도박 등)',
    4: '모호함'
}


def get_image_map(limit=1000):
    image_map = dict()
    for f in glob.glob('images/**', recursive=True):
        if os.path.isdir(f):
            continue
        dirname = os.path.dirname(f)
        basename = os.path.basename(f)
        v = image_map.setdefault(dirname, list())
        if len(v) < limit:
            v.append(basename)
    return image_map


def write_image_map(image_map):
    with open('cache.json', 'w') as f:
        json_val = json.dumps(image_map, indent=4)
        f.write(json_val)


def read_image_map():
    with open('cache.json', 'r') as f:
        json_val = ''.join(f.readlines())
        image_map = json.loads(json_val)
        total_cnt = 0
        for v in image_map.values():
            total_cnt += len(v)
        print(f'Total image count : {total_cnt}')
        return image_map


def show_random_images():
    image_map = read_image_map()
    for i, dirname in enumerate(image_map.keys()):
        if i == 12:
            break
        plt.subplot(3, 4, i+1)
        basename = random.choice(image_map.get(dirname))
        img = Image.open(dirname + '/' + basename)
        img = img.resize((300, 300))
        plt.imshow(np.array(img))
        plt.axis('off')
    plt.show()


def get_images(scaling=True, rescaled_pixels=400):
    """

    :return: images shape -> (n_samples, width, height, channels)
    """
    img_paths, images = [], []
    image_map = read_image_map()
    import tqdm
    for dirname in tqdm.tqdm(image_map.keys()):
        for basename in image_map.get(dirname):
            img_path = dirname + '/' + basename
            img_paths.append(img_path)
            img = Image.open(img_path)
            if scaling:
                img = img.resize((rescaled_pixels, rescaled_pixels))
            img = np.array(img)
            images.append(img)
    images = np.array(images)
    return img_paths, images


if __name__ == "__main__":
    image_map = get_image_map()
    write_image_map(image_map)
    # show_random_images()
