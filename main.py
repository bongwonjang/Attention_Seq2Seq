from preprocessing import *
from attention_module import *

import torch
import torch.nn as nn
from torch import optim 
import nltk.translate.bleu_score as bleu
import os
import time

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, MAX_LENGTH, 
            device):
    ############################################
    #   train (Encoder, Decoder에 맞게 구현해야 함)
    #   
    #   <parameters>
    #   - input_tensor  
    #   - target_tensor  
    #   - encoder                   : Encoder 모듈
    #   - decoder                   : Decoder 모듈
    #   - encoder_optimizer         : Encoder Optim (SGD)
    #   - decoder_optimizer         : Decoder Optim (SGD)  
    #   - criterion                 : Loss 계산 (NLLLoss) 
    #   - MAX_LENGTH                : 문장의 최대 길이
    #   🍦
    #   - device                    : GPU를 사용할 것인지, CPU를 사용할 것인지 선택 
    #
    #    <return>
    #   - encoder                   : Encoder 모듈 (취소)
    #   - decoder                   : Decoder 모듈 (취소)
    #   - loss                      : loss
    ############################################

    SOS_token = 2
    EOS_token = 3

    #####################################
    # encoder와 decoder의 초기 hidden, cell 초기화
    # decoder는 초기 hidden을 encoder에게서 받으므로,
    # 초기화할 필요가 없습니다.
    # 전부 3차원 벡터(?)임을 주의.
    #####################################
    encoder_hidden = encoder.initHidden(device)
    encoder_cell = encoder.initCell(device)

    decoder_hidden = None # useless
    decoder_cell = decoder.initCell(device)

    #####################################
    # 누적된 gradient 초기화
    #####################################
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #####################################
    # 입력 문장의 길이 input_length
    # 타겟 문장의 길이 target_length
    #####################################
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    #####################################
    # 🍪 Pytorch.org의 공식 레퍼런스에 따르면,
    # encoder_outputs를 이용해서 Decoder에게 전달
    # 하지만, hidden vector들을 사용하는 것이 정석이므로 수정합니다.
    #####################################
    # encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    hs_encoder_hiddens = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

    #####################################
    # 출력용 loss 초기화
    #####################################
    loss = 0

    #####################################
    # 🌞
    # for-loop를 통해 Encoder Forward Propagation
    #####################################
    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_cell = encoder(
            input_tensor[ei], encoder_hidden, encoder_cell)

        #####################################
        # encoder_hidden은 (1 * 1 * h)의 3차원 구조의 벡터이다.
        # 따라서, [[[~~~~~]]] 안의 ~~~~~를 얻어내기 위해서
        # encoder_hidden[0, 0]을 통해 끌어낸다(?)
        #####################################
        hs_encoder_hiddens[ei] = encoder_hidden[0, 0]
    
    decoder_input = torch.tensor([SOS_token], device=device)
    
    #####################################
    # 마지막 encoder의 hidden vector가
    # decoder의 초기 입력 hidden vector
    #####################################
    decoder_hidden = encoder_hidden 

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_cell, attn_distr = decoder(
            decoder_input, decoder_hidden, decoder_cell, hs_encoder_hiddens)

        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]  # Teacher forcing

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def print_index_text(index_text, ind2word, country):

    if country == 0: # 소스 언어
        print("English Text----")
    else: # 타깃언어
        print("French Text----")

    for word in index_text:
        print(ind2word[word], end=' ')
    print('')

def convert_index_text(index_text, ind2word):
    result = []
    for word in index_text:
        result.append(ind2word[word])
    result.pop()
    # print(result)
    return result

def evaluate_Iter(input_tensor, target_tensor, encoder, decoder, source_word2index, target_word2index, MAX_LENGTH, device):
    
    SOS_token = 2
    EOS_token = 3

    #########################################
    # hidden vector 및 cell의 크기는 256으로 선택
    #########################################
    hidden_size = 256

    #####################################
    # encoder와 decoder의 초기 hidden, cell 초기화
    # decoder는 초기 hidden을 encoder에게서 받으므로,
    # 초기화할 필요가 없습니다.
    # 전부 3차원 벡터(?)임을 주의.
    #####################################
    encoder_hidden = encoder.initHidden(device)
    encoder_cell = encoder.initCell(device)

    decoder_hidden = None # useless
    decoder_cell = decoder.initCell(device)

    #####################################
    # 입력 문장의 길이 input_length
    # 타겟 문장의 길이 target_length
    #####################################
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    hs_encoder_hiddens = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)

    #####################################
    # 🌞
    # for-loop를 통해 Encoder Forward Propagation
    #####################################
    for ei in range(input_length):
        encoder_output, encoder_hidden, encoder_cell = encoder(
            input_tensor[ei], encoder_hidden, encoder_cell)

        #####################################
        # encoder_hidden은 (1 * 1 * h)의 3차원 구조의 벡터이다.
        # 따라서, [[[~~~~~]]] 안의 ~~~~~를 얻어내기 위해서
        # encoder_hidden[0, 0]을 통해 끌어낸다(?)
        #####################################
        hs_encoder_hiddens[ei] = encoder_hidden[0, 0]
    
    decoder_input = torch.tensor([SOS_token], device=device)
    decoded_text = []
    
    #####################################
    # 마지막 encoder의 hidden vector가
    # decoder의 초기 입력 hidden vector
    #####################################
    decoder_hidden = encoder_hidden 

    for di in range(MAX_LENGTH):
        decoder_output, decoder_hidden, decoder_cell, attn_distr = decoder(
            decoder_input, decoder_hidden, decoder_cell, hs_encoder_hiddens)
        
        output_argmax = torch.argmax(decoder_output, dim=1)
        if output_argmax == 3: # EOS
            decoded_text.append(output_argmax)
            break

        decoded_text.append(output_argmax)

        decoder_input = output_argmax

    return torch.tensor(decoded_text, device=device)

def evaluate(encoder, decoder, source_index2word, source_word2index, target_index2word, target_word2index, MAX_LENGTH, device):
    #########################################
    # tokenizing and preprocessing of test data
    #########################################
    print("tokenizing...")
    source_texts, target_texts, target_labels = tokenize('data/eng-fra_test.txt')

    source_ind_texts = wordtext2indtext(source_texts, source_word2index)
    target_ind_texts = wordtext2indtext(target_texts, target_word2index)
    target_ind_labels = wordtext2indtext(target_labels, target_word2index)

    testing_pairs = [(torch.tensor(s, dtype=torch.long, device=device).view(-1, 1), \
            torch.tensor(t, dtype=torch.long, device=device).view(-1, 1)) \
            for s, t in zip(source_ind_texts, target_ind_labels)]

    #########################################
    # Start testing
    #########################################
    print("\n<START TESTING-------------->")   

    total_bleu = 0

    for iter in range(1, len(testing_pairs) + 1):
        testing_pair  = testing_pairs[iter - 1]
        input_tensor = testing_pair[0]
        target_tensor = testing_pair[1]

        decoded_text = evaluate_Iter(input_tensor, target_tensor, encoder, decoder, source_word2index, target_word2index, MAX_LENGTH, device)

        # print('iter %d ------------------------------' % (iter))
        # print_index_text(input_tensor.squeeze().cpu().tolist(), source_index2word, 0)
        # print_index_text(decoded_text.view(1, -1).cpu().tolist()[0], target_index2word, 1)
        
        bleu_references = (convert_index_text(target_tensor.squeeze().cpu().tolist(), target_index2word))
        bleu_hypotheses = (convert_index_text(decoded_text.view(1, -1).cpu().tolist()[0], target_index2word))

        #########################################
        # BLEU 측정 (nltk 설치 필수)
        #########################################
        total_bleu += bleu.sentence_bleu([bleu_references], bleu_hypotheses)

    print('average BLEU score : %.3f\n' % (total_bleu / len(testing_pairs)))

    return (total_bleu / len(testing_pairs))

def main():
    
    #########################################
    # cuda를 사용할 것인지, cpu를 사용할 것인지 선택
    #########################################
    print("device setting...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("we will use...."+device+"!\n")
    device = torch.device(device)

    #########################################
    # tokenizing and preprocessing of train data
    #########################################
    print("tokenizing...")
    source_texts, target_texts, target_labels = tokenize('data/eng-fra_train.txt')

    source_word2index, source_index2word = preprocess(source_texts)
    target_word2index, target_index2word = preprocess(target_texts)

    source_ind_texts = wordtext2indtext(source_texts, source_word2index)
    target_ind_texts = wordtext2indtext(target_texts, target_word2index)
    target_ind_labels = wordtext2indtext(target_labels, target_word2index)

    source_max_length = max([len(each) for each in source_ind_texts])
    target_max_length = max([len(each) for each in target_ind_texts])
    MAX_LENGTH = max(source_max_length, target_max_length)

    print("------------------------------------")
    print("SOURCE WORD2INDEX MAX LENGTH : ", source_max_length)
    print("TARGET WORD2INDEX MAX LENGTH : ", target_max_length)

    #########################################
    # 학습에 사용할 index화 시킨 영어 문장과
    # 프라스어 문장을 pair로 묶어서 training_pairs에 저장
    #
    # 👍 미리 target_ind_labels으로 pairs에 저장하신 것 탁월했습니다.
    #
    #########################################
    training_pairs = [(torch.tensor(s, dtype=torch.long, device=device).view(-1, 1), \
              torch.tensor(t, dtype=torch.long, device=device).view(-1, 1)) \
             for s, t in zip(source_ind_texts, target_ind_labels)]
    
    #########################################
    # hidden vector 및 cell의 크기는 256으로 선택
    #########################################
    hidden_size = 256

    #########################################
    # encoder와 decoder 객체 생성
    #########################################
    encoder = Encoder(len(source_word2index), hidden_size).to(device)
    decoder = Decoder(len(target_word2index), hidden_size).to(device)
    print("finished making Encoder and Decoder...")

    #########################################
    # encoder와 decoder 각각에 대한 optimizer 생성
    # 순수한 SGD 방식 선택
    #########################################
    learning_rate = 0.01
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    print("finished making Encoder_optimizer and Decoder_optimizer...")

    #########################################
    # Loss function은 Negative Log Likelihood를 사용
    #########################################
    criterion = nn.NLLLoss()
    
    #########################################
    # 학습률 측정에 필요한 변수들
    # - ckpt : 매번 1000번 iteration마다 평균 loss를 출력
    #########################################
    loss_total = 0
    iter_total = 0
    epoches = 50
    
    #########################################
    # Start training
    #########################################
    print("\n<START TRAINING-------------->")    
    encoder.train()
    decoder.train()

    start = time.time()
    
    loss_array = []
    bleu_array = []

    for epoch in range(1, epoches + 1):
        for iter in range(1, len(training_pairs) + 1):
            training_pair  = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            #########################################
            # 🍔 train 함수가 우리가 채울 부분
            #########################################
            loss = train(input_tensor, target_tensor, encoder, 
                        decoder, encoder_optimizer, decoder_optimizer, criterion, 
                        MAX_LENGTH, device)
            
            loss_total += loss 
            iter_total += 1
          
        elapsed = time.time() - start
        # print("입력 문장: {} 기대 출력 문장: {}".format(input_tensor.tolist(), target_tensor.tolist()))
        print('epoch : %d\telapsed time: %.2f min\t avg_loss: %.2f' % (epoch, elapsed / 60, loss_total / iter_total))
        bleu_sc = evaluate(encoder, decoder, source_index2word, source_word2index, target_index2word, target_word2index, 
            MAX_LENGTH, device)

        loss_array.append(loss_total / iter_total)
        bleu_array.append(bleu_sc)

    print('<FINISHED TRAINING-------------->')

    import csv    

    with open("output.csv", "w") as f:
        wr = csv.writer(f)
        wr.writerow(["Epoch", "Loss", "BLEU"])    
        for a in range(0, epoches):
            wr.writerow([a + 1, loss_array[a], bleu_array[a]])
    
def test_only():
    #########################################
    # cuda를 사용할 것인지, cpu를 사용할 것인지 선택
    #########################################
    print("device setting...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("we will use...."+device+"!\n")
    device = torch.device(device)

    #########################################
    # encoder, decoder 불러오기 (⚠️ 불러올 수 있을 때만 사용할 것)
    #########################################
    encoder = torch.load('ENCODER')
    decoder = torch.load('DECODER')
    print("finished loading encoder and decoder as a file")  

    #########################################
    # tokenizing and preprocessing of train data
    #########################################
    print("tokenizing...")
    source_texts, target_texts, target_labels = tokenize('data/eng-fra_train.txt')

    source_word2index, source_index2word = preprocess(source_texts)
    target_word2index, target_index2word = preprocess(target_texts)

    source_ind_texts = wordtext2indtext(source_texts, source_word2index)
    target_ind_texts = wordtext2indtext(target_texts, target_word2index)
    target_ind_labels = wordtext2indtext(target_labels, target_word2index)

    source_max_length = max([len(each) for each in source_ind_texts])
    target_max_length = max([len(each) for each in target_ind_texts])
    MAX_LENGTH = max(source_max_length, target_max_length)

    print("------------------------------------")
    print("SOURCE WORD2INDEX MAX LENGTH : ", source_max_length)
    print("TARGET WORD2INDEX MAX LENGTH : ", target_max_length)

    #########################################
    # 테스트 진행
    #########################################
    with torch.no_grad():
        evaluate(encoder, decoder, source_index2word, source_word2index, target_index2word, target_word2index, 
                MAX_LENGTH, device)

if __name__ == "__main__":
    main()