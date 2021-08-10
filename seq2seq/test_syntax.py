import torch as tor
from torch._C import parse_ir
import torch.nn as nn
embedding=nn.Embedding(num_embeddings=50,embedding_dim=3)
gru=tor.nn.GRU(3,64,2)
a=tor.tensor([1,3,5],dtype=tor.float)
word=tor.tensor([3,5,1])
embedded=embedding(word.view(3,1,-1))
print(embedded.shape)
for x in embedded:
    print(x.shape)
    out,hidden=gru(x,tor.rand((2,1,64)))
print(out.shape)
print(hidden.shape)
log=nn.LogSoftmax(dim=2)
y_h=tor.rand((1,1,13))
y=tor.tensor([12])

loss=nn.NLLLoss()
out=log(y_h)
print(out)
print(loss(out.view(1,-1),y))
print(out.topk(1)[1].item())

