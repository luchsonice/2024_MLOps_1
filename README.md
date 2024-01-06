# MLOps project description
## Overall goal of the project
The goal of this project is to create an image classifier to detect if a person in an image is smoking or not. This is useful for e.g. automatic detection of smokers in smart cities to ensure a clean environment.

This project will also focus on obtaining a well-implemented model pipeline through the use of tools tought in 02476 MLOps at DTU. Such tools are cookiecutter for document structure, docker for reproducible code and Weights and Biases for insight into the performance of the model. 

## Framework that will be used
We will finetune resnet18 from Microsoft Research which is used in many computer vision tasks such as image classification, object detection and segmentation. In order to train the model we will utilize the framework Pytorch Image Models, also known as TIMM. This framework contains several pretrained models and relevant tools for working with resnet in python. Furthermore, we will also use fastai.

## Data
For data we are going to use the dataset Smoker Detection [Image] Classification Dataset from Kaggle https://www.kaggle.com/datasets/sujaykapadnis/smoking. The dataset contains 1120 images split evenly in the two categories, image with smoker and with non-smoker. The dataset has images of smokers from different angles and for the non-smoker they are during similar gestures e.g. a person using an inhaler. The images in the data set are preprocessed and resized to 250x250. The dataset is split into a trainingset 80 % of the images and a testingset 20 % of the images.
