# Nuclei Cell Segmentaion
[![forthebadge](https://forthebadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![forthebadge](https://forthebadge.com/images/badges/built-with-love.svg)](https://forthebadge.com)



## A semantic model for nuclei cell segmentation using U-Net++

## Overview
* [Contributors](#Contributors)
* [Introduction](#Introduction)
* [Dataset Used](#Dataset-Used)
* [Augmentation & Preprocessing](#Augmentation-and-Preprocessing)
* [Network Architecture](#Network-Architecture)
* [Loss Function & Optimizer](#Loss-Function-and-Optimizer)
* [Training setup](#Training-setup)
* [Evaluation Metric](#Evaluation-Metric)
* [Results](#Results)


### Contributors:
This project is created by the joint efforts of
* [Subham Singh](https://github.com/Subham2901)
* [Sandeep Ghosh](https://github.com/Sandeep2017)

### Introduction:
* Cell segmentation is a task of splitting a microscopic image domain into segments,which represent individual instances of cells.It is a fundamental step in many biomedical studies, and it is regarded as cornerstone of image-based cellular research.Cellular morphology is an indicator of a physiological state of the cell, and a well-segmented image can capture biologically relevant morphological information.

* We have tried to create a deep learning algorithm to automate the nucleus detection,the cells’ nuclei is the starting point for most analyses because most of the human body’s 30 trillion cells contain a nucleus full of DNA, the genetic code that programs each cell. Identifying nuclei allows researchers to identify each individual cell in a sample, and by measuring how cells react to various treatments, the researcher can understand the underlying biological processes at work.

### Dataset Used:
The dataset that we have used here is [kaggle 2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018) contest dataset,which contains 670 images of cells along with its masks.
Each image in the dataset is assigned with an average of 20 images that adds up to become the mask of that single image and there are such 670 images in the dataset.Thus,joining the fragmented images into a single mask each time we run the notebook was a hectic task indeed. As a solution, for our own ease we have  saved the created masked files into a separate folder along with the image. Such that we can save the computation time required to join those masks into a single mask eachtime we run the notebook. The link to that folder structure which contains the dataset for training along with separate image for testing  is [provided here.](https://github.com/Subham2901/Nuclei-Cell-segmentaion/tree/master/Data)
#### Some sample images of the dataset is present here:
![](https://github.com/Subham2901/Nuclei-Cell-segmentaion/blob/master/images/dataset.png)


### Augmentation and Preprocessing:
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

Some examples of augmented images and masks are given below.
Augmented Training Images
--------------------------------------------------------------
![]()

Augmented Training Masks
--------------------------------------------------------------![]()

[Back to top](#Nuclei-Cell-Segmentaion)


### Network Architecture:

### Loss Function and Optimizer:

#### Loss Function

#### Optimizer

### Training setup:

### Evaluation Metric:

[Back to top](#Retinal-Vessel-Segmentation)

### Results:

