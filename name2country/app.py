from torch._C import device
from utils import *
from rnn import RNN
name=input("Enter a name: ")
name2tensor=lineToTensor(name).to(device)
rnn=RNN(n_letters,128,n_categories)
rnn.load_state_dict(torch.load("D:/visual studio/my_python/RNN/trained_weight/name2country.pth"))
rnn.eval()
rnn.to(device)
hidden=rnn.create_init_hidden_state().to(device)
for i in range(name2tensor.shape[0]):
    out,hidden=rnn(name2tensor[i],hidden)
print("Calculating...")
print(categoryFromOutput(out)[0])
