#####################################################
# Bong Won Jang's Code
# - 2020 06 15 22:28 ☑️
#####################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

############################################
#   Encoder
#
#   Encoder for seq2seq model with attention mechanism
#   This Encoder is based on a LSTM structure
############################################
class Encoder(nn.Module):

    ############################################
    #   __init__
    #   
    #   <parameters>
    #   - input_size    : the size of input word vocabulary (영어 단어 사전 크기)
    #   - hidden_size   : the size of hidden vector and cell vector
    ############################################
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()

        self.input_size = input_size                                                # scalar : We
        self.hidden_size = hidden_size                                              # scalar : h
        self.cell_size = hidden_size                                                # scalar : h

        self.embedding_matrix = nn.Embedding(self.input_size, self.hidden_size)     # matrix : (We * h)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)

    ############################################
    #   forward
    #   
    #   <parameters>
    #   - word_num  : the integer number of a word (영어 단어 번호)
    #   - hidden    : hidden vector (h_0 is zero vector)
    #   - cell      : cell vector   (c_0 is zero vector)
    #
    #   <return>
    #   - o         : output vector
    #   - hn        : next hidden vector   
    #   - cn        : next cell vector
    ############################################
    def forward(self, word_num, hidden, cell):
        embedding_vector = self.embedding_matrix.weight[word_num].view(1, 1, -1)            #    matrix : (1 * 1 * h)
        o, (hn, cn) = self.lstm(embedding_vector, (hidden, cell))                           #  o matrix : (1 * 1 * h)
                                                                                            # hn matrix : (1 * 1 * h)
                                                                                            # cn matrix : (1 * 1 * h)
        return o, hn, cn

    ############################################
    #   initHidden
    #   
    #   <parameters>
    #   - device     : the integer number of a word
    #
    #   <return>
    #   - initial hidden vector : zero vector
    #
    #   아직 Pytorch 문법에서 3차원으로 구성해야 하는 이유를 모르겠습니다.
    ############################################
    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    ############################################
    #   initCell
    #   
    #   <parameters>
    #   - device     : the integer number of a word
    #
    #   <return>
    #   - initial cell vector : zero vector
    #
    #   아직 Pytorch 문법에서 3차원으로 구성해야 하는 이유를 모르겠습니다.
    ############################################
    def initCell(self, device):
        return torch.zeros(1, 1, self.cell_size, device=device)

############################################
#   Decoder
#
#   Decoder for seq2seq model with attention mechanism
#   This Decoder is based on a LSTM structure
############################################
class Decoder(nn.Module):
    
    ############################################
    #   __init__
    #   
    #   <parameters>
    #   - output_size   : the size of output word vocabulary (프랑스어 단어 사전 크기)
    #   - hidden_size   : the size of hidden vector
    #   - max_length    : the max length of output sentence
    ############################################
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()

        self.output_size = output_size                                              # scalar : Wd
        self.hidden_size = hidden_size                                              # scalar : h
        self.cell_size = hidden_size                                                # scalar : h
        
        self.embedding_matrix = nn.Embedding(self.output_size, self.hidden_size)    # matrix : (Wd * h)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

        self.out_linear = nn.Linear(self.hidden_size * 2, self.output_size)         # eq : (1 * Wd) = (1 * 2h) x (2h * Wd)

    ############################################
    #   forward
    #   
    #   <parameters>                                                       <size>
    #   - word_num  : the integer number of a word (프랑스 단어 번호)    :  scalar
    #   - hidden    : hidden vector                                     :  h 
    #   - cell      : cell vector   (c_0 is zero vector)                :  h
    #   - hs        : pile of all hidden vector from encoder            :  (N * h)
    #
    #   <return>
    #   - o         : output vector
    #   - hn        : next hidden vector   
    #   - cn        : next cell vector
    ############################################
    def forward(self, word_num, hidden, cell, hs):
        embedding_vector = self.embedding_matrix(word_num).view(1, 1, -1)       # matrix : (1 * 1 * h)
        o, (hn, cn) = self.lstm(embedding_vector, (hidden, cell))               #  o matrix : (1 * 1 * h)
                                                                                # hn matrix : (1 * 1 * h)
                                                                                # cn matrix : (1 * 1 * h)               

        attn_score = torch.mm(hs, hn.view(-1, 1)).view(1, -1)                   # (1 * N) = (N * h) x (h * 1) 
        
        attn_distr = F.softmax(attn_score, dim=1)                               # (1 * N) = softmax(1 * N)
        attn_output = torch.mm(attn_distr, hs)                                  # (1 * h) = (1 * N) x (N * h)

        #################################
        # NLLLoss를 사용하기 위해서, Decoder의 y는 log_softmax를 이용해야 한다.
        #################################
        y = F.log_softmax(self.out_linear(torch.cat((attn_output, hn.view(1, -1)), dim=1)), dim=1)  # (1 * output_size)
                                                                                                    # = softmax
                                                                                                    # { (1 * 2h) 
                                                                                                    #      x 
                                                                                                    #   (2h * Wd) }
        return y, hn, cn, attn_distr

    ############################################
    #   initHidden
    #   
    #   <parameters>
    #   - device     : the integer number of a word
    #
    #   <return>
    #   - initial hidden vector : zero vector
    #
    #   아직 Pytorch 문법에서 3차원으로 구성해야 하는 이유를 모르겠습니다.
    ############################################
    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)
        
    ############################################
    #   initCell
    #   
    #   <parameters>
    #   - device     : the integer number of a word
    #
    #   <return>
    #   - initial cell vector : zero vector
    #
    #   아직 Pytorch 문법에서 3차원으로 구성해야 하는 이유를 모르겠습니다.
    ############################################
    def initCell(self, device):
        return torch.zeros(1, 1, self.cell_size, device=device)

