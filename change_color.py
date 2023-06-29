from unet import UNet
from PIL import Image
import torch
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from utils import load_checkpoint
import cv2
from PIL import ImageColor

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default=r'humans\validation\images\img_00000000_1803151818-00000151.jpg', type=str)
parser.add_argument('--color', default='#0a4678')
parser.add_argument('--image_height', default=256, type=int)
parser.add_argument('--image_width', default=256, type=int)

args = parser.parse_args()

def read_image(path):
    image = Image.open(path).convert("RGB")
    image_np = np.array(image)
    return image, image_np

def preprocess_image(image):
    transform = A.Compose([ 
        A.Resize(args.image_height, args.image_width), 
        A.Normalize(), 
        ToTensorV2()
    ])
    image = transform(image=image)
    return image['image']

def predict(image, model, device):
    MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device)
    STD = torch.tensor([0.229, 0.224, 0.225]).to(device)
    model.eval()
    image = image.to(device).unsqueeze(0)
    pred = model(image)
    #applying sigmoid here would also work I guess
    pred = (pred > 0.5).float()
    _, axs = plt.subplots(1, 2)

    image = image * STD[:, None, None] + MEAN[:, None, None]
    # Plot image 1
    axs[0].imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
    axs[0].set_title('Image')

    # Plot image 2
    axs[1].imshow(pred.squeeze().cpu().numpy(), cmap='gray')
    axs[1].set_title('Prediction')
    plt.tight_layout()
    plt.show()

    return pred.squeeze().cpu()
    

def change_color(prediction_mask, color, original_image):
    original_image_c = original_image.copy()
    original_image_c = np.array(original_image_c)
    original_image = np.array(original_image)
    prediction_mask = np.array(prediction_mask)
    original_image[(prediction_mask==0)] = color 
    original_image_w = cv2.addWeighted(original_image, 0.3, original_image_c, 0.7, 0, original_image)
    original_image_c[(prediction_mask==0)] = color
    return original_image_w, original_image_c

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(in_channels=3, out_channels=1).to(device)
    model = load_checkpoint('.\\8_saved_model.pth', model)
    original_image, image = read_image(args.image_path)
    image = preprocess_image(image)
    pred_mask = predict(image, model, device)
    color = ImageColor.getcolor(args.color, "RGB")
    final_image_w, final_image = change_color(pred_mask, color, original_image)

    _, ax = plt.subplots(1, 2)

    ax[0].imshow(final_image_w)
    ax[0].set_title('Background Change 1')

    ax[1].imshow(final_image)
    ax[1].set_title('Background Change 2')

    plt.tight_layout()
    plt.show()   

if __name__=="__main__":
    main()