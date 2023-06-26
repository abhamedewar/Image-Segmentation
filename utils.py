from torch.utils.data import DataLoader
from load_data import HumanDataset
import torch
import os
import time
from tqdm import tqdm
from PIL import Image

def get_loaders(train_image_path, train_mask_path, valid_image_path, valid_mask_path,  \
                transform, batch_size, num_worker, pin_mem):

    train_human_dataset = HumanDataset(train_image_path, train_mask_path, transform = transform)
    validation_human_dataset = HumanDataset(valid_image_path, valid_mask_path, transform = transform)
    
    trainloader = DataLoader(train_human_dataset, batch_size, shuffle=True, num_workers=num_worker, pin_memory=pin_mem)
    validloader = DataLoader(validation_human_dataset, batch_size, shuffle=True, num_workers=num_worker, pin_memory=pin_mem)

    return trainloader, validloader

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint['state_dict'])

def check_accuracy(loader, model, device):
    #implement dice score
    pass

def save_augmentations(train_image_path, train_mask_path, train_transform, s, augment):

    train_aug_image = '.\\train\\augmented_images'
    train_aug_mask = '.\\train\\augmented_mask'

    if not os.path.exists(train_aug_image):
        os.makedirs(train_aug_image)
    
    if not os.path.exists(train_aug_mask):
        os.makedirs(train_aug_mask)

    train_human_dataset = HumanDataset(train_image_path, train_mask_path, augment, transform=train_transform)
    start = time.time()
    loop = tqdm(train_human_dataset, total=len(train_human_dataset), leave=False)
    for img_idx, image, mask in loop:
        image_name = s + '_' + img_idx
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        image.save(os.path.join(train_aug_image, image_name))
        mask.save(os.path.join(train_aug_mask, image_name))
    end = time.time()
    print("Time taken for augmentation:", (end - start)//60, "mins")    
    