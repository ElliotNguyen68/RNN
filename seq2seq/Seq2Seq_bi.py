import torch.nn as nn
import torch
from utils import *
class RNN_en(nn.Module):
    """
       input: input_size=N_in(num words in 1st language),
            GRU: take input shape:
                    1st arg: sq_length,batch_size,dim_embbed
                    2nd arg: num_layer of gru(*2 if bidirection),batch_size,dim embedded
        forward:
            input_sentence:handle a sentence at a time by pass (seq_length,batch_size,embedded_size)->gru
            hidden: init into gru could be None

    """
    def __init__(self,input_size,hidden_size,embedding_size,num_layer) :
        super().__init__()
        self.embedding=nn.Embedding(num_embeddings=input_size,embedding_dim=embedding_size,max_norm=1)
        self.num_gru_layer=num_layer
        self.embedding_size=embedding_size
        self.hidden_size=hidden_size
        self.rnn=nn.GRU(input_size=embedding_size,hidden_size=hidden_size,num_layers=num_layer,bidirectional=True)  

    def forward(self,input_sentence):
        embedded=self.embedding(input_sentence)#(seq_length,1,dim_embedding)
        hidden=torch.zeros((2*self.num_gru_layer,1,self.hidden_size),device=device)
        out,hidden=self.rnn(embedded.view(-1,1,self.embedding_size),hidden)#(seq_length,1,dim_embbeding),(2*num_gru,1,hidden_size)->(seq_length,1,hidden_size*2(bidirectional),(2*gru_layers,1,hiddensize))
        out=out[:,:,:self.hidden_size]+out[:,:,self.hidden_size:]#(seq_length,1,hidden_size)+(seq_lengthm,1,hidden_size)
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
            num_layer in decoder will be twice as in encoder, because using bidirectional in encoder->context(2*num_layer,1,hidden_size)
        forward:
            sentence: a tensor of index word in lang within a sentence, ex: [25,63,102], where each element represent an index in the vocab
            target: just as sentence, but for the target
            train: if True train mode else eval mode
            p: probability of using teacher forcing, [0,1), if >0.5 then use else none
    """  
    def __init__(self,lang_in,lang_out,criterion,hidden_size,num_layer) -> None:
        super().__init__()
        self.encoder=RNN_en(lang_in.n_words,hidden_size,embedding_size=128,num_layer=num_layer)
        self.decoder=RNN_de(lang_out.n_words,hidden_size,embedding_size=128,num_layer=2*num_layer)
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
            loss+=self.criterion(out.view(1,-1),target[0].view(1))
            output.append(self.lang_out.index2word[out.topk(1)[1].item()])
            for i in range(target.shape[0]-1):
                out,hidden=self.decoder(target[i].view(1,1),hidden)
                output.append(self.lang_out.index2word[out.topk(1)[1].item()])
                # print(out.shape)
                loss+=self.criterion(out.view(1,-1),target[i+1].view(1))
        else:
            # use it own output to train 
            out,hidden=self.decoder(torch.tensor([[0]],device=device),hidden)
            loss+=self.criterion(out.view(1,-1),target[0].view(1))
            index=out.topk(1)[1].item()
            output.append(self.lang_out.index2word[index])
            out=torch.tensor([[index]],device=device)
            for i in range(target.shape[0]-1):
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
    rand=random.choice(pairs)
    input,output=tensorFromPair(rand,input_lang,output_lang)
    # num_vocab_in=input_lang.n_words
    # num_vocab_out=output_lang.n_words
    # rnn_en=RNN_en(num_vocab_in,32,9,2).to(device)
    # out,hidden=rnn_en(input)
    # rnn_de=RNN_de(num_vocab_out,32,8,4).to(device=device)
    # in_tensor=torch.tensor([[0]]).to(device)
    # out,hidden=rnn_de(in_tensor,hidden)
    loss=nn.NLLLoss()
    seq2seq=Seq2Seq(lang_in=input_lang,lang_out=output_lang,criterion=loss,hidden_size=32,num_layer=2).to(device)
    seq2seq(input,output,p=1)

    




    