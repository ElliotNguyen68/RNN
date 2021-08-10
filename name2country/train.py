from torch import optim
from torch.nn.modules import loss
from rnn import RNN
from utils import *
import torch
torch.autograd.set_detect_anomaly(True)
def train_1_word(model,optim=None,criterion=None,line_tensor=None,category=None):
    eval=False
    if not optim or not criterion:
        model.eval()
        eval=True
    hidden=model.create_init_hidden_state().to(device)
    for i in range(line_tensor.shape[0]):
        output,hidden=model(line_tensor[i],hidden)
    if eval: return output
    loss=criterion(output,category)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return output,loss.item()
def train(iter,model):
    optim=torch.optim.SGD(model.parameters(),lr=0.005)
    criterion=torch.nn.NLLLoss()
    current_loss=0
    for i in range(iter):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        category_tensor=category_tensor.to(device)
        line_tensor=line_tensor.to(device)
        out,current_loss=train_1_word(model,optim=optim,criterion=criterion,line_tensor=line_tensor,category=category_tensor)
        if i%50==0:
            print(f"{current_loss:4f}")
    torch.save(model.state_dict(),"rnn.pth")
        
def val(model,n_sample):
    corrects=0
    for i in range(n_sample):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        out=train_1_word(model,line_tensor=line_tensor.to(device),category=category_tensor.to(device))
        if categoryFromOutput(out)==category : corrects+=1
    print(f"{corrects}/{n_sample} with {float(corrects)/float(n_sample):4f}")

if __name__=="__main__":
    n_hidden=128

    rnn=RNN(n_letters,n_hidden,n_categories)

    rnn.to(device)
    #this should be 100.000 iterations, but this is enough for learning.
    train(1000,rnn)
    val(rnn,1000)




