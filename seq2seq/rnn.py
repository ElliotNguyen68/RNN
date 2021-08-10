import torch.nn as nn
import torch
from utils import *
class RNN_en(nn.Module):
    """
       input: input_size=N_in(num words in 1st language),
            GRU: take input shape:
                    1st arg: batch,num_word,dim_embbed
                    2nd arg: num_layer of gru,num_word,dim embedded
        forward:
            input_sentence:we still sequentially process each word in sentence shape=(seq_length,1)
                pass into embedding,shape after (seq_length,1,dim_embedding)
            hidden: shape=(num layer in GRU,1,hidden_size)
            embedded :shape=(1,1,dim_embbeding)
            out: shape=(1,1,)    

    """
    def __init__(self,input_size,hidden_size,embedding_size,num_layer) :
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=input_size,embedding_dim=embedding_size,max_norm=1)
        self.rnn=nn.GRU(embedding_size,hidden_size,2)
        self.num_gru_layer=num_layer
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        
    def forward(self,input_sentence):
        embedded=self.embedding(input_sentence)#(seq_length,1,dim_embedding)
        hidden=torch.zeros((self.num_gru_layer,1,self.hidden_size),device=device)
        for x in embedded:
            out,hidden=self.rnn(x.view(1,1,-1),hidden)
        return out,hidden

class RNN_de(nn.Module):
    """
        input: input_size=output_size=N_out(num word in 2nd language)
                embedding_size: dim of embedding
        forward:decoder doesn't have it's own hidden layer , take from encoder's hidden layer
                first iteration: out, hidden=model(hidden, <Start>)
                other: out,hidden=model(out,hidden)
    """
    def __init__(self,input_size,hidden_size,embedding_size,num_layer,output_size=None):
        super().__init__()
        output_size=input_size
        self.embedding=nn.Embedding(input_size,embedding_size,max_norm=1)
        self.rnn=nn.GRU(embedding_size,hidden_size,num_layer)
        self.out=nn.Linear(hidden_size,output_size)
        self.relu=nn.ReLU()
        self.softmax=nn.LogSoftmax(dim=2)
        self.num_gru_layer=num_layer
        self.hidden_size=hidden_size
    def forward(self,input,hidden):
        embedded=self.relu(self.embedding(input))#(1,dim_embedding)
        out,hidden=self.rnn(embedded.view(1,1,-1),hidden.view(self.num_gru_layer,1,self.hidden_size))
        out=self.softmax(self.out(out))
        return out,hidden

class Seq2Seq(nn.Module):
    """
        init:
            lang_in,lang_out:class lang
        forward:
            sentence: a tensor of index word in lang within a sentence, ex: [25,63,102], where each element represent an index in the vocab
            target: just as sentence, but for the target
            train: if True train mode else eval mode
            p: probability of using teacher forcing, [0,1), if >0.5 then use else none
    """  
    def __init__(self,lang_in,lang_out,criterion,hidden_size,num_layer) -> None:
        super().__init__()
        self.encoder=RNN_en(lang_in.n_words,hidden_size,embedding_size=128,num_layer=num_layer)
        self.decoder=RNN_de(lang_out.n_words,hidden_size,embedding_size=128,num_layer=num_layer)
        self.criterion=criterion
        self.optim_en=torch.optim.Adam(self.encoder.parameters(),lr=0.001)
        self.optim_de=torch.optim.Adam(self.decoder.parameters(),lr=0.001)
        self.lang_in=lang_in
        self.lang_out=lang_out
    def forward(self,sentence,target,train=True,p=None):
        if train:
            self.train()
        else: self.eval()
        if not p:
            p=random.random()
            use_teacher_forcing=p>0.5
        else:
            use_teacher_forcing=p>0.5
        _,hidden=self.encoder(sentence)
        loss=0
        output=[]
        if use_teacher_forcing:
            # use target as input to train
            out,hidden=self.decoder(torch.tensor([[0]],device=device),hidden)
            for i in range(1,target.shape[0]-1):
                out,hidden=self.decoder(target[i].view(1,1),hidden)
                output.append(self.lang_out.index2word[out.topk(1)[1].item()])
                # print(out.shape)
                loss+=self.criterion(out.view(1,-1),target[i+1].view(1))
        else:
            # use it own output to train 
            out,hidden=self.decoder(torch.tensor([[0]],device=device),hidden)
            index=out.topk(1)[1].item()
            out=torch.tensor([[index]],device=device)
            for i in range(1,target.shape[0]-1):
                out,hidden=self.decoder(out.view(1,-1),hidden)
                loss+=self.criterion(out.view(1,-1),target[i+1].view(1))
                index=out.topk(1)[1].item()
                output.append(self.lang_out.index2word[index])
                out=torch.tensor([[index]],device=device)

        if train:
            self.optim_de.zero_grad()
            self.optim_en.zero_grad()
            loss.backward()
            self.optim_de.step()
            self.optim_en.step()
            # print(loss)
            return loss,output
        else:
            print(f"{loss:4f}")
            print(f"predict: {' '.join(str(i)for i in output)}")
            

if __name__=="__main__": 
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    rnn_en=RNN_en(input_lang.n_words,1024,9,2).to(device)
    rnn_de=RNN_de(output_lang.n_words,1024,8,2).to(device)
    rand=random.choice(pairs)
    print(rand)
    input,output=tensorFromPair(rand,input_lang,output_lang)
    _,hidden=rnn_en(input.view(input.shape[0],1))
    for x in output:
        out,hidden=rnn_de(x,hidden)
    print(hidden.shape)
    print(out.shape)
    print(output)
    input,output=tensorFromPair(rand,input_lang,output_lang)
    seq2seq=Seq2Seq(input_lang,output_lang,nn.NLLLoss(),1024,2).to(device)
    print(seq2seq(input,output,p=0))



    