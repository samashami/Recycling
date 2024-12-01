import torch
import torch.nn as nn
import torchnet.meter as meter
import matplotlib.pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torchvision import models
from torch.autograd import Variable as V

from avalanche.benchmarks.classic import PermutedMNIST
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive
from avalanche.benchmarks.utils import AvalancheDataset

import numpy as np

from torch.utils.data import DataLoader
from avalanche.models import LeNet5


import torch as t 
from torch.utils import data 
from torchvision import transforms as T 
from PIL import Image  
from torchvision.datasets import ImageFolder
import random
import os
from avalanche.benchmarks import nc_benchmark

class Trash(data.Dataset):
    def __init__(self, root, transform = None, train = True, seed = 1):
        self.labels = {'paper':0, 'glass':1, 'plastic':2, 'cardboard':3, 'trash':4, 'metal':5}
        dirs = os.listdir(root)
        imgs = []
        for d in dirs:
            path = os.path.join(root, d)
            for img in os.listdir(path):
                imgs.append(os.path.join(path, img))

        random.seed(seed)
        random.shuffle(imgs)
        num_train = int(0.7 * len(imgs))
        if train:
            # training
            self.imgs = imgs[:num_train]
        else:
            # validation
            self.imgs = imgs[num_train:]

        if transform is None:    
            normalize = T.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
            if not train:
                # Validation set
                self.transform = T.Compose([
                    T.Resize((32, 32)),
                    # T.CenterCrop(224),
                    T.ToTensor(),
                    normalize
                ])
            else:
                # Training，Augmentation
                self.transform = T.Compose([
                    T.Resize((32, 32)),
                    # T.RandomCrop(224),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = self.labels[img_path.split('/')[-2]]
        data = Image.open(img_path)
        data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.imgs)
    
def plot(losses, accs):
  x = np.arange(len(losses))

  fig, ax1 = plt.subplots()

  color = 'tab:red'
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss', color=color)
  ax1.plot(x, losses, color=color)
  ax1.tick_params(axis='y', labelcolor=color)

  ax2 = ax1.twinx()

  color = 'tab:blue'
  ax2.set_ylabel('Accuracy', color=color)
  ax2.plot(x, accs, color=color)
  ax2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  plt.show()
        
