from __future__ import print_function, division
from builtins import range

import os
import sys
import numpy as np
import pandas as pd

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score
import keras.backend as KK


MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_SIZE = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 5


word2vec = {}
with open(os.path.join('..', '..', '..', 'glove_data', 'glove.6B.{}d.txt'.format(EMBEDDING_SIZE))) as file:
  for line in file:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vector


print('There are {} words'.format(len(word2vec)))

df = pd.read_csv(os.path.join('data', 'train.csv'))

sentences = df['comment_text'].fillna('NA').values
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
targets = df[classes].values

tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)

word2idx = tokenizer.word_index

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor = ', data.shape)

num_words = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_SIZE))
for word, idx in word2idx.items():
  if idx < MAX_VOCAB_SIZE:
    embedding_vector = word2vec.get(word)
    if embedding_vector is not None:
      embedding_matrix[idx] = embedding_vector

embedding_layer = Embedding(
    num_words,
    EMBEDDING_SIZE,
    weights = [embedding_matrix],
    input_length = MAX_SEQUENCE_LENGTH,
    trainable = False
)

input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
x = embedding_layer(input_)
x = LSTM(15, return_sequences=True)(x)
x = GlobalMaxPool1D()(x)
output = Dense(len(classes), activation='sigmoid')(x)

model = Model(input_, output)
model.compile(
    loss = 'binary_crossentropy',
    optimizer=Adam(lr=0.01),
    metrics=['accuracy']
)

n_split = int(len(df) * 0.2)

data_train = data[:n_split,:]
data_test = data[n_split:,:]
target_train = targets[:n_split,:]
target_test = targets[n_split:,:]

r = model.fit(
    data_train,
    target_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
)



p = model.predict(data_test)
auc = []
for i in range(len(classes)):
  auc_class = roc_auc_score(target_test[:, i], p[:, i])
  auc.append(auc_class)

print(np.median(auc))

'''
Train on 25531 samples, validate on 6383 samples
Epoch 1/5
25531/25531 [==============================] - 12s 486us/step - loss: 0.1179 - accuracy: 0.9668 - val_loss: 0.0791 - val_accuracy: 0.9724
Epoch 2/5
25531/25531 [==============================] - 12s 459us/step - loss: 0.0654 - accuracy: 0.9773 - val_loss: 0.0615 - val_accuracy: 0.9782
Epoch 3/5
25531/25531 [==============================] - 12s 459us/step - loss: 0.0592 - accuracy: 0.9793 - val_loss: 0.0620 - val_accuracy: 0.9782
Epoch 4/5
25531/25531 [==============================] - 12s 451us/step - loss: 0.0565 - accuracy: 0.9800 - val_loss: 0.0596 - val_accuracy: 0.9788
Epoch 5/5
25531/25531 [==============================] - 12s 453us/step - loss: 0.0546 - accuracy: 0.9805 - val_loss: 0.0617 - val_accuracy: 0.9787
0.9592728961896806
'''




