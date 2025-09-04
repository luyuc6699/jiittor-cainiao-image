import os

import jittor as jt
from jittor.dataset import Dataset

from PIL import Image
import numpy as np


def load_img(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    img_rgb = np.stack([img_array] * 3, axis=-1)
    return img_rgb


class CustomImageDataset(Dataset):
    def __init__(self, root_dir,
                 metadata=None,
                 augmentations=None,
                 mode_type="train",
                 total_classes=6,
                 **kwargs):
        super().__init__(**kwargs)
        self.root_dir = root_dir
        self.mode_type = mode_type
        self.total_classes = total_classes

        if metadata is not None:
            self.image_info = list(zip(metadata["image_name"], metadata["label"]))
        else:
            self.image_info = list(zip(sorted(os.listdir(root_dir)), [None] * len(os.listdir(root_dir))))

        self.total_len = len(self.image_info)

        self.train_augmentations, self.val_augmentations = augmentations

    def __getitem__(self, index):
        img_filename, label = self.image_info[index]
        img_path = os.path.join(self.root_dir, img_filename)
        img = load_img(img_path)

        if self.mode_type == "train" and self.train_augmentations:
            img_array = self.train_augmentations(image=img)["image"]
            img = np.transpose(img_array, [2,0,1])

        elif self.mode_type == "val" and self.val_augmentations:
            img_array = self.val_augmentations(image=img)["image"]
            img = np.transpose(img_array, [2,0,1])

        if label is not None:
            label_one_hot = np.eye(self.total_classes)[label]
            return jt.array(img), jt.array(label_one_hot)

        return jt.array(img), img_filename
