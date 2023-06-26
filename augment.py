from utils import save_augmentations
import albumentations as A
import cv2

t1 = A.Compose([ 
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.5),
        A.HorizontalFlip(p=0.5),
    ], p=1.0),
])

t2 = A.Compose([
    A.Rotate(limit=40, p=1, border_mode=cv2.BORDER_CONSTANT)
])

def generate_augmentations(train_data_path, train_mask_path, augment):
    for t, s in [(t1, 'combine'), (t2, "rotate")]:
        save_augmentations(train_data_path, train_mask_path, t, s, augment)