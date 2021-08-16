
import torch
import torch.nn as nn
a = torch.rand((5, 1, 3))
b = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
print(a)
print(b)
c = b.unsqueeze(1).unsqueeze(1)*a
print(c)
print(c.sum(dim=0))
embed = nn.Embedding(num_embeddings=5, embedding_dim=3)
test = torch.tensor([3])
print(embed(test))
a = torch.rand((2, 6))
# a = a.permute(1, 0, 2)
b = torch.tensor([0, 1])
print(b.shape)
l = nn.NLLLoss()
print(l(a, b))
