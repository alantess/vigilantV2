import os
import torchvision
import torch
import glob
import cv2
from torch.utils.data import Dataset


class LanesDataset(Dataset):
    def __init__(self, image_path, mask_path, transform=None, dataset_len=1):
        self.transform = transform
        self.d_len = dataset_len
        self.image_path = image_path
        self.img = sorted(glob.glob(os.path.join(image_path, "*.jpg")))
        self.mask = sorted(glob.glob(os.path.join(mask_path, "*.png")))

    def __len__(self):
        return len(self.mask) // self.d_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = self.image_path + str(os.path.basename(
            self.mask[idx]))[:-4] + '.jpg'
        mask = cv2.imread(self.mask[idx], 0)
        img = cv2.imread(img_filename)

        if self.transform:
            normalize = torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))
            img = self.transform(img)
            img = normalize(img)
            mask = self.transform(mask)
            mask = mask * 5
            mask[mask > 1] = 2.0

        return img, mask
