import torch.nn as nn
import torch as tor
from torch.nn.modules.loss import HingeEmbeddingLoss
from torchinfo import summary
from utils import *
class RNN(nn.Module):
    def __init__(self,in_size,hidden_state_size,out_size):
        super().__init__()
        self.in_size=in_size
        self.h_size=hidden_state_size
        self.out_size=out_size
        self.i2h=nn.Linear(in_features=self.in_size+self.h_size,out_features=self.h_size)
        self.i2o=nn.Linear(in_features=self.in_size+self.h_size,out_features=self.out_size)
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,input,hidden):
        combined=tor.cat((input,hidden),dim=1)
        output=self.i2o(combined)
        output=self.softmax(output)
        hidden=self.i2h(combined)
        return output,hidden
    
    def create_init_hidden_state(self):
        #hidden state at first is nothing , just a tensor with random parameters.
        return tor.zeros(1,self.h_size)
    
