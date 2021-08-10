from utils import *
from rnn import RNN

#hidden state size is variable, must try and lot of number to find an appropriate one.
n_hidden=128
input = letterToTensor('A')
hidden = torch.zeros(1, n_hidden)
rnn=RNN(n_letters,n_hidden,n_categories)
rnn.load_state_dict(torch.load("rnn.pth"))
def val(model,n_sample):
    corrects=0
    for i in range(n_sample):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        hidden=torch.zeros(1,n_hidden)
        for i in range(line_tensor.shape[0]):
            out,hidden=model(line_tensor[i],hidden)
        print(categoryFromOutput(out)[0],category) 
        if categoryFromOutput(out)[0]==category : 
            print("hit")
            corrects+=1
    print(f"{corrects}/{n_sample} with {float(corrects)/float(n_sample):4f}")

val(rnn,1000)