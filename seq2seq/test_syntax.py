import torch as tor
from torch._C import parse_ir
import torch.nn as nn
seq_length=5
hidden_size=1024
embdded_size=3
gru_layers=2
gru=nn.GRU(input_size=3,hidden_size=hidden_size,num_layers=gru_layers,bidirectional=True)
a=tor.rand((seq_length,1,embdded_size))
hidden=tor.rand((2*gru_layers,1,hidden_size))
out,hidden=gru(a,None)
print(out.shape)
print(out)
print(hidden.shape)