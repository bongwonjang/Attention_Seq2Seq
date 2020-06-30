#####################################################
# Sung Hyeon Kim's Code (Assisted by Bong Won Jang)
# - 2020 06 15 22:28 ☑️
#####################################################

def tokenize(path_name):

    with open(path_name, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

    source_texts = []
    target_texts = []
    target_labels = []

    for line in lines:
        if not line:
            break
        source_text, target_text = line.split('\t')
        source_text = source_text.strip() 
        target_text = target_text.strip()
                                                                        # -----------Example-----------
        encoder_input = source_text.split()                             # ['come', 'on', '!']
        decoder_input = ("<sos> " + target_text + " <eos>").split()     # ['<sos>', 'allez', '!', '<eos>']
        target_label = (target_text + " <eos>").split()                 # ['allez', '!', '<eos>']

        source_texts.append(encoder_input)
        target_texts.append(decoder_input)
        target_labels.append(target_label)

    return source_texts, target_texts, target_labels

def preprocess(tokenize_texts):
    word2index = {}
    index2word = {}

    #################################################
    # add unk, pad, sos, eos to dictionary in advance
    #################################################

    # word2index
    word2index['<unk>'] = 0
    word2index['<pad>'] = 1
    word2index['<sos>'] = 2
    word2index['<eos>'] = 3
    
    #index2word
    index2word = {v: k for k, v in word2index.items()}

    #################################################
    # add other words to dictionary
    #################################################
    n_word = 4
    for text in tokenize_texts:
        for word in text:
            if word not in word2index:
                word2index[word] = n_word
                index2word[n_word] = word
                n_word += 1

    return word2index, index2word

def wordtext2indtext(word_texts, word2ind):
    ind_texts = []

    for word_text in word_texts:
        temp_ind_text = []
        for word in word_text:
            if word in word2ind:
                temp_ind_text.append(word2ind[word])
            else:
                temp_ind_text.append(word2ind['<unk>'])

        ind_texts.append(temp_ind_text)
        
    return ind_texts

# source_texts, target_texts, target_labels = tokenize('data/eng-fra_test.txt')
# for eng, fra , label_fra in zip(source_texts, target_texts, target_labels):
#     print("(1) ENG :", eng, "\n(2) FRA :", fra, "\n(3) LABEL FRA :", label_fra, "\n")

# source_word2index, source_index2word = preprocess(source_texts)
# target_word2index, target_index2word = preprocess(target_texts)

# print("------------------------------------")
# print("SIZE source_word2index : ", len(source_word2index))
# print(list(source_word2index.items()))
# print(list(source_index2word.items()))

# print("------------------------------------")
# print("SIZE target_word2index : ", len(target_word2index))
# print(list(target_word2index.items()))
# print(list(target_index2word.items()))

# source_ind_texts = wordtext2indtext(source_texts, source_word2index)
# target_ind_texts = wordtext2indtext(target_texts, target_word2index)
# target_ind_labels = wordtext2indtext(target_labels, target_word2index)
# print("------------------------------------")
# for eng, fra , label_fra in zip(source_ind_texts, target_ind_texts, target_ind_labels):
#     print("(1) ENG :", eng, "\n(2) FRA :", fra, "\n(3) LABEL FRA :", label_fra, "\n")