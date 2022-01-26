import torch
import Attention
from util import load_params

params=load_params()
params_conv = params['conv_model']
params_att = params['attention_model']
class attentionModel(torch.nn.Module):
    def __init__(self):
        super(attentionModel,self).__init__()
        self.mpool= torch.nn.MaxPool2d(2)
        self.attention1 = Attention.attentionLayer(params_att['attentionLayer1']['in_channels'],params_att['attentionLayer1']['out_channels'])
        self.attention2 = Attention.attentionLayer(params_att['attentionLayer2']['in_channels'],params_att['attentionLayer2']['out_channels'])
        self.attention3 = Attention.attentionLayer(params_att['attentionLayer3']['in_channels'],params_att['attentionLayer3']['out_channels'])
        self.linear1=torch.nn.Linear(24*3,128)
        self.linear2=torch.nn.Linear(128,64)
        self.linear3=torch.nn.Linear(64,10)
        self.relu= torch.nn.ReLU()
        self.soft= torch.nn.Softmax(dim=-1)

    def forward(self,x):
       
        x=self.mpool(self.relu(self.attention1(x)))
        x=self.mpool(self.relu(self.attention2(x)))
        x=self.mpool(self.relu(self.attention3(x)))
        x=x.view(x.size()[0],-1)
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        x=self.soft(self.linear3(x))
        return x

class convModel(torch.nn.Module):
    def __init__(self):
        super(convModel,self).__init__()
        self.mpool= torch.nn.MaxPool2d(2)
        self.conv1 = torch.nn.Conv2d(params_conv['convLayer1']['in_channels'],params_conv['convLayer1']['out_channels'],3)
        self.conv2 = torch.nn.Conv2d(params_conv['convLayer2']['in_channels'],params_conv['convLayer2']['out_channels'],3)
        self.conv3 = torch.nn.Conv2d(params_conv['convLayer3']['in_channels'],params_conv['convLayer3']['out_channels'],3)
        self.linear1=torch.nn.Linear(8,128)
        self.linear2=torch.nn.Linear(128,64)
        self.linear3=torch.nn.Linear(64,10)
        self.relu= torch.nn.ReLU()
        self.soft= torch.nn.Softmax(dim=-1)

    def forward(self,x):
       
        x=self.mpool(self.relu(self.conv1(x)))
        x=self.mpool(self.relu(self.conv2(x)))
        x=self.mpool(self.relu(self.conv3(x)))
        x=x.view(x.size()[0],-1)
        x=self.relu(self.linear1(x))
        x=self.relu(self.linear2(x))
        x=self.soft(self.linear3(x))
        return x
