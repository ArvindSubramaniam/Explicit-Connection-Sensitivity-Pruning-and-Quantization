
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms


class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

       

        self.convol = nn.Sequential(
                  nn.Conv2d(3, 64, kernel_size= 3, padding=1), 
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(64, 64, kernel_size=3, padding=1), 
                  nn.BatchNorm2d(64),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(64, 128, kernel_size = 3, padding=1), 
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(128, 128, kernel_size = 3, padding=1), 
                  nn.BatchNorm2d(128),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(128, 256, kernel_size = 3, padding=1), 
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 256, kernel_size=3, padding = 1), 
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(256, 256, kernel_size = 3, padding= 1), 
                  nn.BatchNorm2d(256),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(256, 512, kernel_size = 3,padding=1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size = 3, padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size=3 , padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  nn.Conv2d(512, 512, kernel_size = 3, padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size = 3, padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.Conv2d(512, 512, kernel_size = 3, padding = 1), 
                  nn.BatchNorm2d(512),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(kernel_size=2, stride=2),
                  
              )
        
        self.Linear = nn.Sequential(
                  nn.Linear(512, 512),  
                  nn.ReLU(True),
                  nn.BatchNorm1d(512),  
                  nn.Linear(512, 512),
                  nn.ReLU(True),
                  nn.BatchNorm1d(512),  
                  nn.Linear(512, num_classes),)


    def forward(self, x):
        x = self.convol(x)
        x = x.reshape(x.shape[0], -1)
        x = self.Linear(x)  
        x = F.log_softmax(x, dim=1)
        return x

class SimpleResidualBlock(nn.Module):
    def __init__(self, in_channel_size, out_channel_size, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel_size, out_channel_size, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel_size)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel_size, out_channel_size, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel_size)

        if stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv2d(in_channel_size, out_channel_size, kernel_size=1, stride=stride, bias=False)
        self.bn_shortcut= nn.BatchNorm2d(out_channel_size)
        self.relu_shortcut = nn.ReLU(inplace=True)
 

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)
        shortcut= self.bn_shortcut(shortcut)
        
        out = self.relu_shortcut(out + shortcut)
        
        return out

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight, 1.732)
            if m.bias is not None:
                nn.init.zeros_(m.bias)