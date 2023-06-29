import torch
import albumentations as A
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch.optim as optim
from unet import UNet
import argparse
from utils import get_loaders, save_checkpoint, save_augmentations, check_accuracy, load_checkpoint
from augment import generate_augmentations
import wandb
import math
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', default=r'.\humans\train\images', type=str)
parser.add_argument('--train_mask_path', default=r'.\humans\train\masks', type=str)
parser.add_argument('--valid_data_path', default=r'.\humans\validation\images', type=str)
parser.add_argument('--valid_mask_path', default=r'.\humans\validation\masks', type=str)
parser.add_argument('--num_epoch', default=10, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_worker', default=0, type=int)
parser.add_argument('--load_model', default=False, type=bool)
parser.add_argument('--image_height', default=256, type=int)
parser.add_argument('--image_width', default=256, type=int)
parser.add_argument('--save_checkpoints', default=False, type=str)
parser.add_argument('--num_classes', default=1, type=int)
parser.add_argument('--amp', default=False, type=int)
parser.add_argument('--augment', default=False, type=bool)


args = parser.parse_args()

learning_rate = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using ", device)
wandb.init(project="Semantic Segmentation")

def train(loader, valid_loader, model, optimizer, loss_fn, scaler, num_epoch, epoch, scheduler):
    n_steps_per_epoch = math.ceil(len(loader.dataset) / args.batch_size)
    loop = tqdm(enumerate(loader), total=len(loader), leave=False)
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

        metrics = {"train/train_loss": loss.item(),"train/epoch":  (idx + (n_steps_per_epoch * epoch)) / n_steps_per_epoch}
        wandb.log({**metrics})
        if idx%700 == 0:
            if idx == 0:
                continue
            valid_loss, pixel_acc, dice_score = check_accuracy(valid_loader, model, loss_fn, device, True, 3)
            val_metrics = {"val/val_loss": valid_loss, "val/pixel_accuracy": pixel_acc, "val/dice_score": dice_score}
            wandb.log({**val_metrics})

    scheduler.step()
    print(f"Epoch ({epoch}/{num_epoch}]")
    print(f"Loss {loss.item()}")


# def run_prediction(model, valid_loaderr, loss): 
#     MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
#     STD = torch.tensor([0.229, 0.224, 0.225]).to(device)

#     # valid_loss, pixel_acc, dice_score = check_accuracy(valid_loaderr, model, loss, device, True, 3)
#     # print(valid_loss, pixel_acc, dice_score)
#     model.eval()

#     with torch.no_grad():
#         loop = tqdm(enumerate(valid_loaderr), total=len(valid_loaderr), leave=False)
#         i = 0
#         for idx, (data, target) in loop:
#             data   = data.to(device)
#             target  = target.float().to(device).unsqueeze(1)
#             pred = torch.sigmoid(model(data))
#             pred = (pred > 0.5).float()
#             data = data * STD[:, None, None] + MEAN[:, None, None]
#             fig, axs = plt.subplots(1, 3)

#             # Plot image 1
#             axs[0].imshow(target.squeeze().cpu().numpy(), cmap='gray')
#             axs[0].set_title('Target')

#             # Plot image 2
#             axs[1].imshow(pred.squeeze().cpu().numpy(), cmap='gray')
#             axs[1].set_title('Prediction')

#             # Plot image 3

#             axs[2].imshow(data.squeeze().permute(1, 2, 0).cpu().numpy())
#             axs[2].set_title('Image')

#             # Adjust spacing and display the plot
#             plt.tight_layout()
#             plt.show()
#             i += 1
#             if i == 10:
#                 exit(0)

def main():
    
    transform = A.Compose([ 
        A.Resize(args.image_height, args.image_width), 
        A.Normalize(), 
        ToTensorV2()
    ])

    if args.augment:
        generate_augmentations(args.train_data_path, args.train_mask_path, args.augment)
    
    model = UNet(in_channels=3, out_channels=1).to(device)

    if args.load_model:
        model = load_checkpoint('.\\saved_model_89.08.pth', model)

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
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    grad_scaler = torch.cuda.amp.GradScaler()
    loss = nn.CrossEntropyLoss() if args.num_classes > 1 else nn.BCEWithLogitsLoss()

    # run_prediction(model, valid_loader, loss)
    # exit()

    for epoch in range(args.num_epoch):
        train(train_loader, valid_loader, model, optimizer, loss, grad_scaler, args.num_epoch, epoch, scheduler)
        valid_loss, pixel_acc, dice_score = check_accuracy(valid_loader, model, loss, device, True, 3)
        val_metrics = {"val/val_loss_epoch": valid_loss, "val/pixel_accuracy_epoch": pixel_acc, "val/dice_score_epoch": dice_score}
        wandb.log({**val_metrics})
        if epoch % 2 == 0:
            checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch":epoch}
            name = str(epoch) + '_' +  "saved_model.pth"
            save_checkpoint(checkpoint, name)
    wandb.finish()


if __name__=="__main__":
    main()
    print("....Completed....")


