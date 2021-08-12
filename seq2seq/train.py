from utils import *
# from Seq2Seq import *
import random
from Seq2Seq_bi import *

def train(epochs,iterations,model,pairs,lang_in,lang_out):
    for epoch in range(epochs):
        print(f"Epoch {epoch}:")
        for i in range(iterations):
            pair=random.choice(pairs)
            input,output=tensorFromPair(pair,lang_in,lang_out)
            model(input,output)
        input,output=tensorFromPair(pair,lang_in,lang_out)
        model(input,output,train=False)
        print(f"actual sentence: {pair[1]}")
        torch.save(model.state_dict(),"rnn.pth")
if __name__=="__main__":
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    loss=nn.NLLLoss()
    rnn=Seq2Seq(input_lang,output_lang,loss,256,2).to(device)
    train(100,1,rnn,pairs,input_lang,output_lang)
