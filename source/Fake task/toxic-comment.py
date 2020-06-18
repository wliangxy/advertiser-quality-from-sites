from __future__ import print_function, division
from builtins import range


import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from sklearn.metrics import roc_auc_score


#tokenizer = Tokenizer(num_words=20)
#sentences = "vahid sanei"
#tokenizer.fit_on_texts(sentences)
#sequences = tokenizer.texts_to_sequences(sentences)
#print(sequences)
#exit(0)


MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

word2vec = {}


print('Loading the data ...')
with open(os.path.join('..','..','..','glove_data','glove.6B.{}d.txt'.format(EMBEDDING_DIM))) as f:
  for line in f:
    arr = line.split()
    word = arr[0]
    embedding = np.asarray(arr[1:], dtype='float32')
    word2vec[word] = embedding

print('There has been {} words in the glove dataset'.format(len(word2vec)))
print('Example: {}: embedding = {}'.format('apple', word2vec['apple']))


train = pd.read_csv(os.path.join('data', 'train.csv'))
sentences = train['comment_text'].fillna('NA').values

classes  = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
targets = train[classes].values

#print(targets[:10])


tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

print("sequences = ", sequences[:10])



word2idx = tokenizer.word_index
print('Number of tokens = {}'.format(len(word2idx)))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor = ', data.shape)





















