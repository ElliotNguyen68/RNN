from utils import *
from rnn import RNN
import torch.nn as nn
def train_1_word(model,optim,criterion,line,line_tensor,category_tensor):
    model.train()
    hidden=model.create_init_hidden()
    loss=0
    target_word=target(line).unsqueeze(-1)
    for i in range(line_tensor.shape[0]-1):
        out,hidden=model(category_tensor,line_tensor[i],hidden)
        loss+=criterion(out,target_word[i+1])
    optim.zero_grad()
    loss.backward()
    optim.step()
    return out,loss.item()

def eval(model,line,line_tensor,category_tensor,criterion):
    model.eval()
    hidden=model.create_init_hidden()
    res=[]
    loss=0
    # print(line_tensor.shape)
    # print(category_tensor.shape)
    target_word=target(line).unsqueeze(-1)
    # print(target_word)
    for i in range(line_tensor.shape[0]-1):
        out,hidden=model(category_tensor,line_tensor[i],hidden)
        # print(out.shape)
        # print(target_word[i+1])
        loss+=criterion(out,target_word[i+1])
        res.append(wordFromOutput(out)[0])
    return res,loss

def train(interations):
    rnn=RNN(n_categories,n_letters,128)
    optim=torch.optim.Adam(rnn.parameters(),lr=0.0005)
    criterion=nn.NLLLoss()
    for x in range(interations):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        if x%1000==0:
            out,loss=eval(rnn,line,line_tensor,category_tensor,criterion)
            print(f"epoch: {x}, loss={loss}, predict: {line[0]}{''.join(str(i) for i in out)}, actual: {line}")
            continue
        out,loss=train_1_word(rnn,optim,criterion,line,line_tensor,category_tensor)
    torch.save(rnn.state_dict(),"rnn(w2w).pth")
    
if __name__=="__main__":
    train(1000)
