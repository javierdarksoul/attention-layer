import sys
sys.path.append("src/")
import torchvision
from train import train
from model import attentionModel
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np


mnist_train_data = torchvision.datasets.MNIST('datasets/', train=True, download=True,
                                              transform=torchvision.transforms.ToTensor())
mnist_test_data = torchvision.datasets.MNIST('datasets/', train=False, download=True,
                                         transform=torchvision.transforms.ToTensor())

np.random.seed(1)
idx = list(range(len(mnist_test_data)))
np.random.shuffle(idx)
split = int(0.7*len(idx))

train_loader = DataLoader(mnist_train_data, batch_size=64, 
                          sampler=SubsetRandomSampler(idx[:split]))

valid_loader = DataLoader(mnist_train_data, batch_size=64, 
                          sampler=SubsetRandomSampler(idx[split:]))

test_loader= DataLoader(mnist_test_data, batch_size=64)


model = attentionModel()
train(model,train_loader,valid_loader, 10, 1e-3, True ,"trained_models/attention_model")

