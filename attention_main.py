import sys
import json
sys.path.append("src/")
import torchvision
from train import train
from model import attentionModel
from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np
from util import load_params
import torch
from sklearn.metrics import precision_recall_fscore_support

params=load_params()

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

test_loader= DataLoader(mnist_test_data, batch_size=1)#len(mnist_test_data))


model = attentionModel()
train(model,train_loader,valid_loader, params['train']['epoch'], params['train']['learnrate'], params['train']['usecuda'] ,params['train']['pathsave'] , "metrics/metrics_attention.json")


model.load_state_dict(torch.load(params['train']['pathsave'] + "/best.pt"))
model= model.cpu()
predictions=[]
labels=[]
for image, label in test_loader:
    #print(image.shape)
    prediction = model(image)
    
    predictions.append(prediction.argmax())
    labels.append(label)

labels=torch.tensor(labels)
predictions=torch.tensor(predictions)
kpi= precision_recall_fscore_support(labels.detach().numpy(), predictions.detach().numpy(), average=None)


kpi_dict={'class 0': kpi[0][0],'class 1': kpi[0][1],'class 2': kpi[0][2],'class 3': kpi[0][3],'class 4': kpi[0][4],'class 5': kpi[0][5],'class 6': kpi[0][6] ,'class 7': kpi[0][7],'class 8': kpi[0][8],'class 9': kpi[0][9]}
with open("metrics/precision_attention.json", "w") as f:
            f.write(json.dumps(kpi_dict))

kpi_dict={'class 0': kpi[1][0],'class 1': kpi[1][1],'class 2': kpi[1][2],'class 3': kpi[1][3],'class 4': kpi[1][4],'class 5': kpi[1][5],'class 6': kpi[1][6] ,'class 7': kpi[1][7],'class 8': kpi[1][8],'class 9': kpi[1][9]}
with open("metrics/recall_attention.json", "w") as f:
            f.write(json.dumps(kpi_dict))


kpi_dict={'class 0': kpi[2][0],'class 1': kpi[2][1],'class 2': kpi[2][2],'class 3': kpi[2][3],'class 4': kpi[2][4],'class 5': kpi[2][5],'class 6': kpi[2][6] ,'class 7': kpi[2][7],'class 8': kpi[2][8],'class 9': kpi[2][9]}
with open("metrics/f1_score_attention.json", "w") as f:
            f.write(json.dumps(kpi_dict))