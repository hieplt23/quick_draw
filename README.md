<div align="center">
  <img src="https://emojipedia-us.s3.amazonaws.com/source/skype/289/artist-palette_1f3a8.png" width="100" height="100"/>
</div>

<h1 align="center">Quick Draw!🖌️</h1>
<p align="center">
  <strong>A neural network-based drawing classification project using Google's Quick, Draw! dataset.</strong>
</p>

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Categories](#categories)
- [Training](#training)
- [Experiments](#experiments)
- [Demo](#demo)
- [Requirements](#requirements)

## Introduction
The **Quick Draw** project uses Google's *Quick, Draw!* dataset, featuring millions of hand-drawn sketches. A neural network model classifies doodles into **20** categories, each with **10,000** images. After **20** epochs of training, the model achieved a test loss of **0.37** and an accuracy of **89.9%**. The project culminates in a real-time drawing app built with **OpenCV**, providing instant predictions for user sketches.

## Dataset 
You can view and download the data from the following link: [QuickDraw Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn)

## Categories
Below is the list of the 20 labels used to train the model:

|           |           |           |           |
|-----------|:-----------:|:-----------:|:-----------:|
|   apple   |   book    |   bowtie  |   candle  |
|   cloud   |    cup    |   door    | envelope  |
|eyeglasses |  guitar   |   hammer  |    hat    |
| ice cream |   leaf    | scissors  |   star    |
|  t-shirt  |   pants   | lightning |    tree   |

## Training
To train the model, you need to download the `.npz` files corresponding to the 20 classes used and store them in the **data** folder. If you want to train your model with a different list of categories, you only need to change the constant **CLASSES** in `./src/config.py` and download the necessary `.npz` files. Then, simply run: `python train.py`

## Experiments
For each class, I selected the first 10,000 images and split them into training and test sets with a ratio of 8:2. The training and test loss/accuracy curves for the experiment are shown below:

<p align="center">
  <img src="demo/loss_accuracy_curves.png" alt="Loss and Accuracy Curves" width="700">
</p>

## Demo
By run script: `painting_app.py`
<p align="center">
 <img src="./demo/demo1.gif" width=800>
</p>

## Requirements
* **python 3.12**
* **cv2 4.10**
* **pytorch 2.3.1** 
* **numpy**
