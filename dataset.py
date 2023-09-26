import os
import cv2
import torch
import random
import numpy as np
from torch.utils.data import Dataset


class FoodDataset(Dataset):

    def __init__(self, mode, transforms=None, img_size=(564, 564)):

        super().__init__()
        self.class_to_int = {'hd': 0, 'nhd': 1}
        self.mode = mode
        self.dir = 'data/train' if self.mode == 'train' else 'data/test'
        self.imgs = self._get_filenames()
        self.transforms = transforms
        self.img_size = img_size

    def _get_filenames(self):
        return os.listdir(self.dir)

    def __getitem__(self, idx):

        image_name = self.imgs[idx]

        # Reading, converting and normalizing image
        img = cv2.imread(self.dir + '/' + image_name, cv2.IMREAD_COLOR)
        # img = cv2.resize(img, self.img_size)
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
            img /= 255.
            img = torch.from_numpy(img).permute(2, 0, 1)
        except:
            print(self.dir + '/' + image_name)

        if self.mode == "train" or self.mode == "val":

            label = self.class_to_int[image_name.split("_")[0]]
            label = torch.tensor(label, dtype=torch.float32)

            if self.transforms:
                img = self.transforms(img)

            return img, label

        elif self.mode == "test":
            # Apply Transforms on image
            if self.transforms:
                img = self.transforms(img)
            return img

    def __len__(self):
        return len(self.imgs)


def read_img_test(path, device):
    # Reading, converting and normalizing image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (564, 564))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img /= 255.
    img = img[..., np.newaxis]
    img = torch.from_numpy(img).permute(3, 2, 0, 1)
    img.to(device)
    return img


def read_img_augment(path):
    # Reading, converting and normalizing image
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (564, 564))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return img


def read_random_images(num, path):
    files = random.sample(os.listdir(path), num)
    lst = []
    for file in files:
        path = f'{path}/{file}'
        lst.append(read_img_augment(path))
    images = np.array(lst, dtype=np.uint8)
    return images
