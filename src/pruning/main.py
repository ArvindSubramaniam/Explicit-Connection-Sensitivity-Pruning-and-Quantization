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
from models import *
from ECS import *

def network_init():
    
    
  model = VGG16([ 64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
       512, 512, 512, 'M'
   ])
  # model = resnet34().cuda()
  # model = AlexNet()
  optimiser = optim.SGD( model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
  scheduler = optim.lr_scheduler.StepLR(optimiser, lr_decay_interval, gamma=0.1)


  train_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010)),
  ])

  test_transform = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
      transforms.ToTensor(),
      transforms.Normalize((0.4914, 0.4822, 0.4465),
                          (0.2023, 0.1994, 0.2010)),
  ])

  train_dataset = CIFAR10('_dataset', True, train_transform, download=True)
  test_dataset = CIFAR10('_dataset', False, test_transform, download=False)

  train_loader = DataLoader( train_dataset, batch_size, shuffle=True, num_workers=2, pin_memory=True)
  val_loader = DataLoader( test_dataset, batch_size, shuffle=False, num_workers=2, pin_memory=True)

  return model, optimiser, scheduler, train_loader, val_loader


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
