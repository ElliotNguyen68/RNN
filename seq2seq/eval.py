from utils import *
from Seq2Seq_bi import *

def eval(epochs,model,pairs,lang_in,lang_out):
    for epoch in range(epochs):
        print(f"Time: {epoch}:")
        pair=random.choice(pairs)    
        input,output=tensorFromPair(pair,lang_in,lang_out)
        model(input,output,train=False,p=0)
        print(f"actual sentence: {pair[1]}")
def find_best_translation(epochs,model,pairs,lang_in,lang_out):
    for epoch in range(epochs):
        print(f"Time: {epoch}:")
        pair=random.choice(pairs)    
        input,output=tensorFromPair(pair,lang_in,lang_out)
        model.decode_beam_search(input,num_k=5)
        model(input,output,train=False,p=0)
        print(f"actual sentence: {pair[1]}")

if __name__=="__main__":
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    loss=nn.NLLLoss()
    rnn=Seq2Seq(input_lang,output_lang,loss,256,2).to(device)
    rnn.load_state_dict(torch.load("seq2seq/Seq2Seq_bi.pth"))
    find_best_translation(100,rnn,pairs,input_lang,output_lang)