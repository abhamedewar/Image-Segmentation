# Image Segmentation- U-Net architecture from scratch

Link to the paper: https://arxiv.org/abs/1505.04597

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



