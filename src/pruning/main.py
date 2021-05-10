import torch
from train import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision import transforms
import numpy as np
import copy
import os


model, optimiser, lr_scheduler, train_loader, val_loader = network_init()
model = model.to(device)
density = 5
criterion = nn.CrossEntropyLoss()

train_loss = training(1, model, optimiser, lr_scheduler, criterion, device,train_loader)
val_loss, val_acc = validate(1, model, criterion, device, val_loader)
keep_masks = pruning(model, density)  
apply_prune_mask(model, keep_masks)




if __name__ == '__main__':

      max = 0
      path = 'net.pt'
      path2 = 'AlexNet.pt'
      for epoch in range(epochs):

          if os.path.exists(path):
              checkpoint = torch.load(path)
              model.load_state_dict(checkpoint['state_dict'])

          train_loss = training(epoch, model, optimiser, lr_scheduler, criterion, device,train_loader)
          val_loss, val_acc = validate(epoch, model, criterion, device, val_loader)

          

          lr_scheduler.step()

          keep_masks = pruning(model, density)  
          # apply_prune_mask(model, keep_masks)

          if max < val_acc*100:
              torch.save({'state_dict': model.state_dict()}, path)

          print('Epoch: {} \t train-Loss: {:.4f}, \tval-Loss: {:.4f}'.format(epoch+1,  train_loss, val_loss))
          print(f'Validation Accuracy: {round(val_acc*100,2)}')
