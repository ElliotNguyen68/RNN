from torch.autograd.grad_mode import F
import torch.nn as nn
import torch
from utils import *
torch.autograd.set_detect_anomaly(True)


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

    def __init__(self, input_size, hidden_size, embedding_size, num_layer):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=input_size, embedding_dim=embedding_size, max_norm=1)
        self.num_gru_layer = num_layer
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layer, bidirectional=True)

    def forward(self, input_sentence):
        """"
            gru(input,hidden)
                input: entire sentence
                hidden: could be None
            -> out,hidden:
                out: contains hidden states of of time step
                hidden: contains only final hidden state of last word in the sentence
        """
        embedded = self.embedding(input_sentence)
        hidden = None
        # (seq_length,1,dim_embbeding),(2*num_gru,1,hidden_size)->(seq_length,1,hidden_size*2(bidirectional),(2*gru_layers,1,hiddensize))
        embedded.unsqueeze_(1)  # (seq_length,1,dim_embbeding)
        out, hidden = self.rnn(embedded, hidden)
        # (seq_length,1,hidden_size)+(seq_lengthm,1,hidden_size)
        out = out[:, :, :self.hidden_size]+out[:, :, self.hidden_size:]
        # hidden_sum = hidden.view(self.num_gru_layer, 2, self.hidden_size)
        # hidden_sum = hidden[:, 0, :]+hidden[:, 1, :]
        # hidden_sum = torch.sum(hidden, dim=0)
        return out, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size, num_gru_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.energy = nn.Linear(hidden_size*2, hidden_size)
        self.weight = nn.Linear(hidden_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=0)
        self.num_gru = num_gru_layers
        self.relu = nn.ReLU()

    def forward(self, prev_de_hidden, en_hiddens):
        """

        Args:
            prev_de_hidden ([type]): hidden state of step t-1 of the decoder, shape:(1,1,hidden_size)
            en_hiddens ([type]): hidden state of entire input sentence ,shape:(seq_length,1,hidden_size)
        Math: 
            repeat prev decoder hidden state seq_length time 
            concat with encoder hiddens state 
            pass through a Linear function to find score of similarity (2*hidden_size,hidden_size) => (seq_length,1,hidden_size)
            Make it to non-linear by using a Tanh funciton (-1,1)
            pass through a 1-Linear function to weigth for each sentences =>seq_length,1,1
            Make it to probability by pass through a softmax function

        Returns:
            probability of attention at step t in decoder should pay on which hidden state in the encoder, base on hidden state of the decoder at time t-1
        """

        prev_de_hidden = torch.sum(prev_de_hidden, dim=0)
        seq_length = en_hiddens.shape[0]
        hidden_repeated = prev_de_hidden.repeat(seq_length, 1, 1)
        combined = torch.cat((hidden_repeated, en_hiddens), dim=2)
        scores = self.tanh(self.energy(combined))
        # print("scores", scores.shape)
        w = self.softmax(self.weight(scores))
        # print("w", w.shape)

        return w


class RNN_de(nn.Module):
    """
        input: input_size=output_size=N_out(num word in 2nd language)
                embedding_size: dim of embedding
        forward:decoder doesn't have it's own hidden layer , take from encoder's hidden layer
                first iteration: out, hidden=model(hidden, <Start>)
                other: out,hidden=model(out,hidden)
    """

    def __init__(self, input_size, hidden_size, embedding_size, num_layer, output_size=None):
        super().__init__()
        output_size = input_size
        self.embedding = nn.Embedding(input_size, embedding_size, max_norm=1)
        self.rnn = nn.GRU(embedding_size+hidden_size, hidden_size, num_layer)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.attention = Attention(
            hidden_size=hidden_size, num_gru_layers=num_layer)
        self.num_gru_layer = num_layer
        self.hidden_size = hidden_size

    def forward(self, input, hidden, encoder_outputs):
        """feed forward pass in decoder

        Args:
            input ([type]): contains input word(dim_embedding)
            hidden ([type]): hidden state of up till prev word
            encoder_ouputs ([type]): entire hidden states of input sentences
        Math:
            get the attention of hidden relative to encoder_outputs
            compute context and concat it with input
        Returns:
            predicted word at time step t
        """
        embedded = self.embedding(input).unsqueeze(0)
        # print(embedded.shape)
        match = self.attention(hidden, encoder_outputs)  # (seq_length,1,1)
        scores_en_hiddens = encoder_outputs*match
        # print(scores_en_hiddens.shape)
        context = torch.sum(scores_en_hiddens, dim=0,
                            keepdim=True)  # (1,1,hidden_size)
        # print(context.shape)
        # (1,1,hidden_size+embedded_size)
        combined = torch.cat((context, embedded), dim=2)
        out, hidden = self.rnn(combined, hidden)
        out = self.softmax(self.out(out))
        return out, hidden


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

    def __init__(self, lang_in, lang_out, criterion, hidden_size, num_layer) -> None:
        super().__init__()
        self.encoder = RNN_en(lang_in.n_words, hidden_size,
                              embedding_size=128, num_layer=num_layer)
        self.decoder = RNN_de(lang_out.n_words, hidden_size,
                              embedding_size=128, num_layer=2*num_layer)

        self.criterion = criterion
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
        self.lang_in = lang_in
        self.lang_out = lang_out

    def forward(self, src, target, train=True, p=None):
        if train:
            self.train()
        else:
            self.eval()
        if not p:
            p = random.random()
            use_teacher_forcing = p > 0.75
        else:
            use_teacher_forcing = p > 0.75
        target_seq_length = target.shape[0]
        target_vocab_size = self.lang_out.n_words
        # tensor to store decoder outputs
        outputs = torch.zeros(target_seq_length, 1,
                              target_vocab_size).to(device)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        input = torch.tensor([0], device=device)
        loss = 0
        if train == False:
            ans_greedy = []
        for t in range(0, target_seq_length-1):

            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(
                input.view(1), hidden, encoder_outputs)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # get the highest predicted token from our predictions
            val, index = output.topk(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = target[t] if use_teacher_forcing else torch.tensor(
                [index.item()]).to(device)
            try:
                ans_greedy.append(index.item())
            except:
                pass
            # print("input", input.shape)
            # print("top1", top1)

        # print(outputs.shape)
        # print(target.shape)
        loss += self.criterion(outputs.squeeze(1), target)
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            # print(loss)
            return loss, outputs
        else:
            print(f"{loss:4f}")
            print(
                f"greedy search: {' '.join(str(self.lang_out.index2word[i])for i in ans_greedy )}")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # only in eval, to find best sentence
    def decode_beam_search(self, input_sentence, num_k=2):
        def clear(a, length):
            for i, x in enumerate(a):
                # remove all element in list have current length < length
                if len(x.arr) < length and len(a) > num_k:
                    a.pop(i)  # and make sure enough num_k element remaining
        self.eval()
        topk = []  # contain top k least negative score
        encoder_outputs, hidden = self.encoder(input_sentence)
        print(encoder_outputs.shape)
        SOS = thread_sentence("SOS", -10, hidden)
        x = SOS
        word = x.arr[-1]
        hidden_word = x.hidden
        input = torch.tensor([self.lang_out.word2index[word]], device=device)
        out, hidden_word = self.decoder(
            input, hidden_word.detach(), encoder_outputs)
        scores, index = out.topk(num_k)
        scores = scores.view(num_k)
        index = index.view(num_k)
        for score, i in zip(scores, index):
            current_thread = thread_sentence(
                self.lang_out.index2word[i.item()], score.item(), hidden_word)
            current_thread.append_head(x)
            topk.append(current_thread)
        step = 0
        limit_height = 10000
        all_EOS = True
        lenght_current = 1
        ended_sentences = []
        while step < limit_height:
            x = topk[step % num_k]
        # for id,x in enumerate(topk):
            word = x.arr[-1]
            hidden_word = x.hidden
            if word not in ['.', '!', '?', 'EOS']:
                input = torch.tensor(
                    [self.lang_out.word2index[word]], device=device)
                out, hidden_word = self.decoder(
                    input, hidden_word, encoder_outputs)
                scores, index = out.topk(num_k)
                scores = scores.view(num_k)
                index = index.view(num_k)
                for score, i in zip(scores, index):
                    current_thread = thread_sentence(
                        self.lang_out.index2word[i.item()], score.item(), hidden_word)
                    current_thread.append_head(x)
                    lenght_current = max(
                        lenght_current, len(current_thread.arr))
                    # print(current_thread.arr)
                    topk.append(current_thread)

                # topk.pop(id)
                all_EOS = False
            else:
                ended_sentences.append(x)
            # topk.sort(key=lambda x: x.score,reverse=True)
            step += 1
            if step > 0 and (step) % num_k == 0:
                clear(topk, lenght_current)
                topk.sort(key=lambda x: x.score, reverse=True)
                topk = topk[:num_k]
                # print(id,len(topk))
            if all_EOS or step > limit_height:
                break
        ended_sentences.sort(key=lambda x: x.score, reverse=True)
        print(f"beam search: {' '.join(ended_sentences[0].arr) }")


class thread_sentence:
    def __init__(self, word, current_score, hidden) -> None:
        self.score = current_score
        self.arr = [word]
        self.hidden = hidden

    def append_head(self, old_thread):
        tmp = old_thread.arr[:]
        tmp.extend(self.arr)
        self.arr = tmp
        step = len(self.arr)
        self.score += old_thread.score*(step-1)
        self.score /= step


if __name__ == "__main__":
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    rnn = Seq2Seq(input_lang, output_lang, nn.NLLLoss(), 128, 2)
    print(rnn.count_parameters())
    rand = random.choice(pairs)
    input, output = tensorFromPair(rand, input_lang, output_lang)
    num_vocab_in = input_lang.n_words
    num_vocab_out = output_lang.n_words
    # rnn_en = RNN_en(num_vocab_in, 3, 9, 2).to(device)
    # # print(rand)
    # out, hidden = rnn_en(input)
    # print(out.shape)
    # print(hidden.shape)
    # attention = Attention(3).to(device)
    # print(attention(hidden, out))
    # rnn_de = RNN_de(num_vocab_out, 3, 8, 2).to(device=device)
    # in_tensor = torch.tensor([0]).to(device)

    # out, hidden = rnn_de(in_tensor, hidden)

    loss = nn.NLLLoss()
    seq2seq = Seq2Seq(lang_in=input_lang, lang_out=output_lang,
                      criterion=loss, hidden_size=128, num_layer=2).to(device)
    print(seq2seq)
    print(seq2seq.count_parameters())
    print(input.shape)
    print(output.shape)
    loss, out = seq2seq(input, output, train=True, p=1)
    loss, out = seq2seq(input, output, train=True, p=0)
    print(loss)
    print(out.shape)
    seq2seq.decode_beam_search(input, num_k=3)
