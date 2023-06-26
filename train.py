import torch
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from unet import UNet
import argparse
from utils import get_loaders, save_checkpoint, save_augmentations
from augment import generate_augmentations

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', default=r'.\data_split\train\images', type=str)
parser.add_argument('--train_mask_path', default=r'.\data_split\train\masks', type=str)
parser.add_argument('--valid_data_path', default=r'.\data_split\validation\images', type=str)
parser.add_argument('--valid_mask_path', default=r'.\data_split\validation\masks', type=str)
parser.add_argument('--num_epoch', default=2, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_worker', default=0, type=int)
parser.add_argument('--load_model', default=False, type=bool)
parser.add_argument('--image_height', default=600, type=int)
parser.add_argument('--image_width', default=600, type=int)
parser.add_argument('--save_checkpoints', default=False, type=str)
parser.add_argument('--num_classes', default=1, type=int)
parser.add_argument('--amp', default=False, type=int)
parser.add_argument('--augment', default=False, type=bool)


args = parser.parse_args()

learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(loader, model, optimizer, loss_fn, scaler, num_epoch):
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for epoch in range(num_epoch):
        for idx, (data, targets) in loop:
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            with torch.cuda.amp.autocast():
                pred = model(data)
                loss = loss_fn(pred, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loop.set_description(f"Epoch ({epoch}/{num_epoch}]")
            loop.set_postfix(loss = loss.item())
    

def main():
    
    transform = A.Compose([ 
        A.Resize(args.image_height, args.image_width), 
        A.Normalize(), 
        ToTensorV2()
    ])

    if args.augment:
        generate_augmentations(args.train_data_path, args.train_mask_path, args.augment)

    model = UNet(in_channels=3, out_channels=1).to(device)
    train_loader, valid_loader = get_loaders(
        args.train_data_path, 
        args.train_mask_path, 
        args.valid_data_path,
        args.valid_mask_path,
        transform,  
        args.batch_size, 
        num_worker=args.num_worker,
        pin_mem=False
        )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler()
    loss = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss()
    # train(train_loader, model, optimizer, loss, grad_scaler, args.num_epoch)

    # checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    # save_checkpoint = save_checkpoint(checkpoint, "checkpoint.pth.tar")

if __name__=="__main__":
    main()
    print("....Completed....")


