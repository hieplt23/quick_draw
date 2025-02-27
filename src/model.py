import torch.nn as nn
from math import pow
import torch

class QuickDrawModel(nn.Module):
    def __init__(self, input_size=28, num_classes=20):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, bias=False), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))

        dimension = int(64 * pow(input_size / 4 - 3, 2))
        self.fc1 = nn.Sequential(nn.Linear(dimension, 512), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(512, 128), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.conv1(input)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
