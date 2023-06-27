from torch.utils.data import DataLoader
from load_data import HumanDataset
import torch
import os
import time
from tqdm import tqdm
from PIL import Image
import wandb
import torchvision.transforms as T
import torchvision
import matplotlib.pyplot as plt

def get_loaders(train_image_path, train_mask_path, valid_image_path, valid_mask_path,  \
                transform, batch_size, num_worker, pin_mem):

    train_human_dataset = HumanDataset(train_image_path, train_mask_path, transform = transform)
    validation_human_dataset = HumanDataset(valid_image_path, valid_mask_path, transform = transform)
    
    trainloader = DataLoader(train_human_dataset, batch_size, shuffle=True, num_workers=num_worker, pin_memory=pin_mem)
    validloader = DataLoader(validation_human_dataset, batch_size, shuffle=True, num_workers=num_worker, pin_memory=pin_mem)

    return trainloader, validloader

def save_checkpoint(state, filename):
    torch.save(state, filename)

def load_checkpoint(chk_path, model):
    checkpoint = torch.load(chk_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def check_accuracy(loader, model, loss_fn, device, log_images=False, batch_idx=0):

    '''
    Intersection:

    (preds * mask).sum(): This computes the element-wise multiplication of the predicted mask (preds)
    and the ground truth mask (mask). The resulting tensor will have a value of 1 (True) only where both
    masks have a positive value (overlap).

    .sum(): The sum operation calculates the total number of pixels where the intersection occurs (number of True values).
    Union:

    (preds + mask).sum(): This computes the element-wise addition of the predicted mask (preds) and the
    ground truth mask (mask). The resulting tensor will have a value of 1 (True) where either of the masks
    has a positive value.

    .sum(): The sum operation calculates the total number of pixels where the union occurs (number of True values).
    Epsilon:

    1e-7: A small constant added to the denominator to avoid division by zero.
    It ensures numerical stability in case both the intersection and union are zero.
    
    '''

    num_correct = 0
    num_pixels  = 0
    dice_score  = 0

    model.eval()

    with torch.no_grad():
        loop = tqdm(enumerate(loader), total=len(loader), leave=False)
        for idx, (data, target) in loop:
            data   = data.to(device)
            target  = target.float().to(device).unsqueeze(1)
            pred = model(data)
            loss = loss_fn(pred, target)
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()
            num_correct += (pred == target).sum()
            num_pixels += torch.numel(pred)
            dice_score += (2 * (pred * target).sum()) / (
                (pred + target).sum() + 1e-7
            )
        
            if idx == batch_idx and log_images:
                log_image_table(data, pred, target, device) 

    pixel_accuracy = (num_correct/num_pixels) * 100
    dice = dice_score/len(loader)*100
    print(f"Got {num_correct}/{num_pixels} with pixel accuracy {pixel_accuracy:.2f}")
    print(f"Dice score: {dice:.2f}")

    return loss.item(), pixel_accuracy, dice

def log_image_table(images, masks, target, device):

    MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).to(device)
    table = wandb.Table(columns=["image", "predicted mask", "target"])
    for img, pred, target in zip(images, masks, target):
        pred = pred * 255
        target = target * 255
        img = img * STD[:, None, None] + MEAN[:, None, None]
        table.add_data(wandb.Image(img.squeeze().permute(1, 2, 0).cpu().numpy()), 
                       wandb.Image(pred.squeeze().cpu().numpy()), wandb.Image(target.squeeze().cpu().numpy()))
    wandb.log({"predictions_table":table}, commit=False)

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
