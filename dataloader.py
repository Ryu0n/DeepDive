import numpy as np
from glob import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from config import *


config = Config()


class SpamImageDataset(Dataset):
    def __init__(self, is_train: bool, train_ratio=0.7):
        self.is_train = is_train

        self.train_image_list = list()
        self.test_image_list = list()
        for n_class in range(1, 4):
            image_paths_per_class = glob(f'spam_images/{n_class}/*')
            criterion = int(len(image_paths_per_class) * train_ratio)
            for i, image_path in enumerate(image_paths_per_class):
                if i < criterion:
                    self.train_image_list.append((n_class, image_path))
                else:
                    self.test_image_list.append((n_class, image_path))

    def __len__(self):
        return len(self.train_image_list if self.is_train else self.test_image_list)

    def __getitem__(self, index):
        image_list = self.train_image_list if self.is_train else self.test_image_list
        n_class, image_path = image_list[index]
        image = np.array(Image.open(image_path).convert("RGB").resize((config.img_size, config.img_size))).transpose((2, 0, 1))
        return n_class, image


def load_dataloader():
    return map(lambda dataset: DataLoader(dataset, batch_size=config.batch_size, shuffle=True),
               (SpamImageDataset(is_train=True),
                SpamImageDataset(is_train=False))
               )
