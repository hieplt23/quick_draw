<p align="center">
 <h1 align="center">Quick Draw! 🖌️🖼</h1>
</p>

## Introduction
The Quick Draw project leverages Google's Quick, Draw! dataset, which features millions of hand-drawn sketches across a wide range of categories. The project involves implementing a neural network model to classify these doodles into 20 randomly selected categories, each containing 10,000 images. The model was trained over 20 epochs, achieving a test loss of 0.37 and a test accuracy of 89.9%. The end result is a real-time drawing application built with OpenCV that allows users to draw and instantly receive predictions on their sketches.

## Dataset 
[QuickDraw Dataset]
You can view and download the data from the following link: https://console.cloud.google.com/storage/browser/quickdraw_dataset/sketchrnn

## Categories:
The table below lists the 20 labels that I used to train the model.

|           |           |           |           |
|-----------|:-----------:|:-----------:|:-----------:|
|   apple   |   book    |   bowtie  |   candle  |
|   cloud   |    cup    |   door    | envelope  |
|eyeglasses |  guitar   |   hammer  |    hat    |
| ice cream |   leaf    | scissors  |   star    |
|  t-shirt  |   pants   | lightning |    tree   |

## Training
You need to download npz files corresponding to 20 classes my model used and store them in folder **data**. If you want to train your model with different list of categories, you only need to change the constant **CLASSES** at **src/config.py** and download necessary npz files. Then you could simply run **python train.py**

## Experiments:
For each class, I take the first 10000 images, and then split them to training and test sets with ratio 8:2. The training/test loss/accuracy curves for the experiment are shown below:

<img src="demo/loss_accuracy_curves.png" width="800"> 

## Demo
<img src="./demo/demo.gif" width=800>

## Requirements
* **python 3.12**
* **cv2**
* **pytorch** 
* **numpy**
