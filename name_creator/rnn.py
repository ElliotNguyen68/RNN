import torch
import torch.nn as nn
from utils import *
class RNN(nn.Module):
    def  __init__(self,n_categories,input_size,hidden_size):
        super().__init__()
        output_size=input_size#one word to one word
        self.hidden_size=hidden_size
        self.i2o=nn.Linear(in_features= input_size+hidden_size+n_categories,out_features= output_size)
        self.i2h=nn.Linear(in_features= input_size+hidden_size+n_categories,out_features= hidden_size)
        self.o2o=nn.Linear(in_features= hidden_size+output_size,out_features= output_size)
        self.drop=nn.Dropout(p=0.1)
        self.softmax=nn.LogSoftmax(dim=1)

    def forward(self,category,input,hidden):
        # print(category.shape)
        # print(input.shape)
        # print(hidden.shape)
        combined=torch.cat((category,input,hidden),dim=1)
        # print(combined.shape)
        output=self.i2o(combined)
        next_hidden=self.i2h(combined)
        out_combined=torch.cat((output,hidden),dim=1)
        out=self.softmax(self.drop(self.o2o(out_combined))) 
        return out,next_hidden

    def create_init_hidden(self):
        return torch.zeros((1,self.hidden_size))

if __name__=="__main__":
    rnn=RNN(n_categories,n_letters,128)
    print(rnn)
    category, line, category_tensor, line_tensor = randomTrainingExample()
    # print(category_tensor.shape)
    print(line_tensor.shape)
    out,hidden=(rnn(category_tensor,torch.rand((1,59)),rnn.create_init_hidden()))
    print(out.shape)
    print(hidden.shape)


        
        