import torch as tor
import torch.nn as nn

in_h=tor.tensor([[4,3,2],[4,6,8]],dtype=tor.float32)

print(in_h.topk(1))

sof=nn.Softmax(dim=1)

logsof=nn.LogSoftmax(dim=1)

print(logsof(in_h))

a=tor.rand((1,17))
b=tor.rand((1,12))
c=tor.rand((1,65))
d=tor.cat((a,b,c),dim=1)
print(d.shape)
