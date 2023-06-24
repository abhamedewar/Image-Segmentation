import torch
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from unet import UNet
import argparse
from utils import get_loaders, save_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=r'.\data\images', type=str)
parser.add_argument('--mask_path', default=r'.\data\masks', type=str)
parser.add_argument('--epoch', default=2, type=int)
parser.add_argument('--batch_size', default=2, type=int)
# parser.add_argument('--num_worker', default=0, type=int)
parser.add_argument('--load_model', default=False, type=bool)
parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)
parser.add_argument('--save_checkpoints', default=False, type=str)
parser.add_argument('--num_classes', default=1, type=int)
parser.add_argument('--amp', default=False, type=int)


args = parser.parse_args()

learning_rate = 1e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(loader, model, optimizer, loss, scaler, num_epoch):
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
    for epoch in num_epoch:
        for idx, (data, targets) in loop:
            data = data.to(device)
            targets = targets.float().unsqueeze(1).to(device)

            with torch.cuda.amp.autocast():
                pred = model(data)
                loss = loss(pred, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        loop.set_description(f"Epoch ({epoch}/{num_epoch}]")
        loop.set_postfix(loss = loss.item())
    

def main():

    train_transform = A.Compose([
        A.Resize(args.image_height, args.image_width), 
        A.HorizontalFlip(p=0.6),
        A.Normalize(), 
        ToTensorV2()
    ])

    valid_transform = A.Compose([
        A.Resize(args.image_height, args.image_width), 
        A.Normalize(), 
        ToTensorV2()
    ])

    model = UNet(in_channels=3, out_channels=1)
    train_loader, valid_loader = get_loaders(
        args.data_path, 
        args.mask_path, 
        train_transform, 
        valid_transform, 
        args.batch_size, 
        num_worker=0,
        pin_mem=False
        )
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    grad_scaler = torch.cuda.amp.GradScaler()
    loss = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss()
    train(train_loader, model, optimizer, loss, grad_scaler, args.num_epoch)

    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint = save_checkpoint(checkpoint, "checkpoint.pth.tar")

if __name__=="__main__":
    main()
    print("....Completed....")


