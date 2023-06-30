# Deep Background: U-Net-Based Image Background Segmentation and Editing

Enhancing Images with Custom Backgrounds
## $\color{blue}{UNet-\ Implementation\ from\ scratch}$
## $\color{blue}{Dice Score: 98.90}$

### Project Summary

This project focuses on background segmentation and color modification in images using a U-Net model. The U-Net architecture, is implemented from scratch to accurately segment the background from an input image.

The goal of the project is to provide a solution for changing the background color of images while preserving the main subject. The trained U-Net model generates a mask that identifies the background region in an image. This mask is then used to modify the background color, allowing for the incorporation of custom backgrounds.

https://github.com/abhamedewar/Image-Segmentation/assets/20626950/077220c6-02c8-4784-94e4-2b9c96e01b81

### Output Screenshot

<img src="https://github.com/abhamedewar/Image-Segmentation/assets/20626950/c5675b68-c24d-4c9e-95d5-4f197f3fe589" width="400" height="400">

<img src="https://github.com/abhamedewar/Image-Segmentation/assets/20626950/7148784d-f588-4929-9301-53d50e253706" width="400" height="400">

### Human Segmentation Dataset

This dataset is a collection of images used for human segmentation.

### Dataset Details

* Number of Training Images: 24,042
* Number of Validation and Test Images: 10,383
* Image Size: 256 x 256 pixels

### Running the training code:

```
python train.py --train_data_path <folder with all images> --train_mask_path <folder with train masks>
--valid_data_path <folder with validation images> --valid_mask_path <folder with validation masks>
```

Refer train.py for to change other parameters

### Running the prediction code:

```
python change_color.py ----image_path <path to image to change color> --color <hex color value>
```


### Model Architecture:

The architecture used for semantic segmentation is __U-Net, which is implemented from scratch.__

### Data Preprocessing:

The following data preprocessing techniques were applied to the training dataset:

* Horizontal Flip: Randomly flips the images horizontally to augment the dataset.
* Rotation: Randomly rotates the images to add variability.
* Random Blur

Albumentation library was used to perform augmentation of training images and masks.

### Training Process

The model was trained using the following configurations:

* Number of Epochs: 10
* Learning Rate: 0.0001
* Learning Rate Scheduler: Step LR
* Batch Size: 16

### Validation Metric
The validation of the segmentation model was evaluated using the following metrics:

* __Dice Score:__ Measures the overlap between the predicted segmentation mask and the ground truth mask.
* __Pixel Accuracy:__ Computes the accuracy of the predicted segmentation at the pixel level.

To visualize the training process and monitor the model's performance, Weight and Biases (wandb) was used. The following visualizations were tracked:

* Training Loss: The loss value during training.
* Validation Loss: The loss value during validation.
* Validation Accuracy: The accuracy of the model on the validation dataset.
* Sample Images: Selected images from the training dataset were displayed to observe the model's predictions during training.
  
By leveraging the U-Net architecture, applying data preprocessing techniques, and monitoring the training process with visualization, the semantic segmentation model was trained and evaluated effectively.

### Results:

__The Dice Score obtained for the semantic segmentation model: 98.90%__


Validation Dice Score and Pixel Accuracy Plots:

![image](https://github.com/abhamedewar/Image-Segmentation/assets/20626950/5f90e656-554a-4859-bcb7-18787c928219)

Training Loss:

![image](https://github.com/abhamedewar/Image-Segmentation/assets/20626950/b6caf2b9-2e04-422d-a1d2-57fbd3f76c71)


Validation Loss:

![image](https://github.com/abhamedewar/Image-Segmentation/assets/20626950/4d19409e-e802-4363-821a-85794b5d6814)

Predictions:

![image](https://github.com/abhamedewar/Image-Segmentation/assets/20626950/bccfe40c-fda4-44d0-9d6a-801251bf360f)

### Other details
Image segmentation is the process of dividing an image into multiple distinct regions or segments based on the characteristics of the pixel in the image. 

Now that we know that each pixel is classified into some class and pixels of same class are grouped together, segmentation can be further classified into the following types:

1. Semantic Segmentation
2. Instance Segmentation
3. Panoptic Segmentation

<img src="https://github.com/abhamedewar/Image-Segmentation/assets/20626950/1dfc1f78-3026-4c8c-bec8-524edabcd7f1" width="500" height="400">

Source: https://www.v7labs.com/blog/image-segmentation-guide

### Architecture of U-Net:

<img src="https://github.com/abhamedewar/Image-Segmentation/assets/20626950/5f314f21-a13e-45e8-b0a9-270e3e9bcaf9" width="600" height="450">

1. Crop the image when applying skip connection.
2. The input size and the output size does not match. The input image is padded.

So, why skip connections??

Downsampling causes loss of spatial information so during the upsampling phase high resolution features are passed in each upsample layer. This helps in better classification of each pixel of the input image.



