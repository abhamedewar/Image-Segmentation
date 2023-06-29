from torch.utils.data import Dataset
import torch
from PIL import Image
import os
import numpy as np
import torchvision.transforms.functional as F

class HumanDataset(Dataset):

    def __init__(self, image_dir, mask_dir, augment=False, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def preprocess(self, img):
        # img[img == 255.0] = 1.0
        img = (img > 127)
        return img

    def __getitem__(self, index):
        image_id = os.path.join(self.image_dir, self.images[index])
        mask_id = os.path.join(self.mask_dir, self.images[index]).replace('img', 'mask')
        #might have to convert these to np.array
        image = np.array(Image.open(image_id).convert("RGB"))
        mask = np.array(Image.open(mask_id).convert("L"), dtype=np.float32)            #some preprocessing might be required here
        if not self.augment:
            mask = self.preprocess(mask)

        if self.transform:
            augmentations = self.transform(image = image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']

        if self.augment:
            return image_id.split('\\')[-1], image, mask
        
        return image, mask
    