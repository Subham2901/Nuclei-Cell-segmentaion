# Nuclei Cell Segmentaion
-----------------------------------------------------------------------------------------------------------------------------------------
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)



## A semantic segmentation model for nucleus cell segmentation using U-Net++

## Overview
-----------------------------------------------------------------------------------------------------------------------------------------
* [Contributors](#Contributors)
* [Introduction](#Introduction)
* [Dataset Used](#Dataset-Used)
* [Augmentation & Preprocessing](#Augmentation-and-Preprocessing)
* [Network Architecture](#Network-Architecture)
* [Loss Function & Optimizer](#Loss-Function-and-Optimizer)
* [Learning Rate](#Learning-Rate)
* [Training setup](#Training-setup)
* [Evaluation Metric](#Evaluation-Metric)
* [Results](#Results)


### Contributors:
-----------------------------------------------------------------------------------------------------------------------------------------
This project is created by the joint efforts of
* [Subham Singh](https://github.com/Subham2901)
* [Sandeep Ghosh](https://github.com/Sandeep2017)

### Introduction:
-----------------------------------------------------------------------------------------------------------------------------------------
* Cell segmentation is a task of splitting a microscopic image domain into segments,which represent individual instances of cells.It is a fundamental step in many biomedical studies, and it is regarded as cornerstone of image-based cellular research.Cellular morphology is an indicator of a physiological state of the cell, and a well-segmented image can capture biologically relevant morphological information.

* We have tried to create a deep learning algorithm to automate the nucleus detection,the cells’ nuclei is the starting point for most analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. Identifying nuclei allows researchers to identify each individual cell in a sample, and by measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work.

### Dataset Used:
-----------------------------------------------------------------------------------------------------------------------------------------
The dataset that we have used here is [kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) contest dataset,which contains 670 images of cells along with its masks.
Each image in the dataset is assigned with an average of 20 images that adds up to become the mask of that single image and there are such 670 images in the dataset.Thus,joining the fragmented images into a single mask each time we run the notebook was a hectic task indeed. As a solution, for our own ease we have  saved the created masked files into a separate folder along with the image. Such that we can save the computation time required to join those masks into a single mask eachtime we run the notebook. The link to that folder structure which contains the dataset for training along with separate image for testing  is [provided here.](https://github.com/Subham2901/Nuclei-Cell-segmentaion/tree/master/Data)
#### Some sample images along with its masks is present here:

![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/Image().JPG)


#### Masks:
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/mask().JPG)


### Augmentation and Preprocessing:
-----------------------------------------------------------------------------------------------------------------------------------------
The training data was augmented on the fly using the [Albumentations library](https://albumentations.ai/).
A strong combination of different types of image augmentations were applied with varied probabilities. They were:
* CLAHE.
* Rotate.
* Flip.
* GaussNoise.
* HorizontalFlip.
* VerticalFlip.
* HueSaturationValue.
* Random Gamma.
* Random Brightness & contrast.

Along with the above mentioned augmentations, every image in the training and testing sets underwent a Histogram Equalization preprocessing step, i.e, CLAHE (Contrast Limited Adaptive Histogram Equalization).

##### Augmented Training Images:
-----------------------------------------------------------------------------------------------------------------------------------------
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/Aug_img.JPG)

##### Augmented Masks:
-----------------------------------------------------------------------------------------------------------------------------------------
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/aug_mask.JPG)


[Back to top](#Nuclei-Cell-Segmentaion)


### Network Architecture:
-----------------------------------------------------------------------------------------------------------------------------------------
When it comes to medical imaging, the margin of error is almost negligible. One small mistake can lead to even fatal outcomes. Thus, algorithms designed for medical imaging must achive high performannce and accuracy even if the total no. of training samples upon which the data is trained is not enough or satisfactory,to solve this issue ,[UNet](https://arxiv.org/abs/1505.04597) was introduced in the world of AI based medical imaging world, and it gave unexpetedly good results.The __model architecture__ that we have used here is known as [UNet++](https://arxiv.org/abs/1807.10165).The UNet++ architecture has three additions in comparison to the classic UNet:
* __Redesigned skip pathways -__ These are added to bridge the semantic gap between the encoder and the decoder subpaths.As a result, the optimiser can optimise it more easily.
* __Dense skip connections -__ The Dense blocks used here is adopted from the DenseNet with the purpose to improve the segmentation accuracy and improves the gradient flow.

* __Deep supervision -__ The soul purpose of the deep supervison is to maintain the balance between the speed(inference) and perpormance of the model as per our requirements. There are mainly two modes that deep supervison has:
* Accurate Mode-__ In this case the output from all the segmentation branches are averaged.
* Fast Mode -__ In this mode the  final segmentation map is selected on the basis of prediction metric from one of the segmentation block.
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/UNET%2B%2B(gless).JPG)
### Loss Function and Optimizer:
-----------------------------------------------------------------------------------------------------------------------------------------
The loss function and the optimizer that we have used here are BCE DICE LOSE and ADAM AND BEYOND respectively.
#### Loss Function-

The loss function that we have used here is a combination of both [Binary Cross Entropy](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy) loss function and [Dice](https://arxiv.org/abs/1606.04797) loss function.

```Python
def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = y_true * y_pred
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    
```
[Back to top](#Nuclei-Cell-Segmentaion)
#### Optimizer:-
The optimizer that we have used here to optimize our model is [ADAM and Beyond](https://arxiv.org/abs/1904.09237?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%253A+arxiv%252FQSXk+%2528ExcitingAds%2521+cs+updates+on+arXiv.org%2529). Which uses a new exponential moving average AMSGRAD. The AMSGRAD uses a smaller learning rate in comparison to ADAM. In case of ADAM the decrement or decay of learning rate is not guaranteed where as AMSGRAD  uses smaller learning rates , it maintains the maximum of  all the learning rates until the present time step and uses that maximum value for normalizing the running average of the gradient instead of learning rate in ADAM or RMSPROP. Thus, it converges better than ADAM or RMSPROP

#### Learning Rate:
-----------------------------------------------------------------------------------------------------------------------------------------
The learning rate we have used here is not constant throughout the training of the data, instead we have used a learning rate schedular, which increases/decreases the learning rate gradually after every fixed set of epochs such that  we can attain the optimum convergence by the end of our training of the data.

### Training setup:
-----------------------------------------------------------------------------------------------------------------------------------------
* GPU: Nvidia P100 16GB
* CPU: Intel Xeon
* RAM: 12GB DDR4

The network was trained using the above mentioned setup for  epochs with a batch size of ```10``` and input image size ```256 x 256 x 3```. Total time taken for training is 15 mins.
#### Training VS Validation Accuracy Curve:
-----------------------------------------------------------------------------------------------------------------------------------------
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/acc_resized.JPG)
#### Training VS Validation Loss Curve:
-----------------------------------------------------------------------------------------------------------------------------------------
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/loss.JPG)
-----------------------------------------------------------------------------------------------------------------------------------------
[Back to top](#Nuclei-Cell-Segmentaion)
### Evaluation Metric:
[F1-Score](https://en.wikipedia.org/wiki/F1_score) was used along with [IOU](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU#:~:text=IOU%20is%20defined%20as%20follows,of%200%20to%20mask%20values.) and [AUC](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC) for the evaluation of the results.

-----------------------------------------------------------------------------------------------------------------------------------------

[Back to top](#Nuclei-Cell-Segmentaion)

### Results:
The following table shows the results that we have achieved:
F1-Score | AUC | IoU|
--- | --- | --- |
89.0 | 93.0 | 81.0|
##### Images:
-----------------------------------------------------------------------------------------------------------------------------------------
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/P_img.JPG)
#### Masks:
-----------------------------------------------------------------------------------------------------------------------------------------
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/p_masks.JPG)

#### Predicted Masks:
-----------------------------------------------------------------------------------------------------------------------------------------
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/pp_resized.JPG)



>Keep Coding!! :)


