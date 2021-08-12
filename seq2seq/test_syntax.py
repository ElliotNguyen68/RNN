from inspect import trace
from pickle import TRUE
import torch
import torch.nn as nn
a=torch.rand((1,1,10))
log=nn.LogSoftmax(dim=2)
out=log(a)
score,index=a.topk(3)
print(score.view(3))
print(index.view(3))
a=[1,2,4]
a.sort(reverse=True)
b=[3,4,1]
for x,y in zip(a,b):
    print(x+y)
