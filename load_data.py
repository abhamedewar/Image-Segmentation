from torch.utils.data import Dataset
import torch
from PIL import Image
import os

class HumanDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def preprocess(self, img):
        img[img == 255.0] = 1.0
        return img

    def __getitem__(self, index):
        image_id = os.path.join(self.image_dir, self.images[index])
        mask_id = os.path.join(self.mask_dir, self.images[index])
        #might have to convert these to np.array
        image = Image.open(image_id)
        mask = Image.open(mask_id)            #some preprocessing might be required here

        mask = self.preprocess(mask)

        if self.transform:
            augmentations = self.transform(image)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask
    