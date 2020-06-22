from glove_embeddings import word2vec

import os
import sys
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras.models import Model
from keras.layers import Embedding, Dense, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from sklearn.metrics import roc_auc_score

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')
  if sys.version_info > (3.0):
    os.system('python3 -m nltk.downloader stopwords')
  else:
    os.system('pyhton -m nltk.downloader.stopwords')

MAX_SEQ_LEN = 500
MAX_VOC_SIZE = 20000
EMBEDDING_SIZE = 100
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10
STOPS = set(stopwords.words('english'))

def preprocess(text):
  '''
    1) we remove stop words
    2) we remove words with length 1
    3) all words are converted to lowercase
  '''
  arr = word_tokenize(text)
  return [word.lower() for word in arr if len(word) > 1 and not word in STOPS]

df = pd.read_csv(os.path.join('..','..','yelp_data','updated','business.csv'))

#print(df.columns)
'''
Index(['business_id', 'name', 'address', 'city', 'state', 'postal_code',
       'latitude', 'longitude', 'stars', 'review_count', 'is_open',
       'attributes', 'categories', 'hours', 'url', 'is_eng', 'webpage_text'],
      dtype='object')
'''
classes = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
#starts (classes/labels) = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}

#convert column stars to one hot encoding
df['stars'] = pd.Categorical(df['stars'], categories=classes)
dummies = pd.get_dummies(df['stars'])
df = pd.concat([df, dummies], axis=1)
df.drop('stars', axis=1, inplace=True)
'''
CategoricalIndex([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], categories=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, ...], ordered=False, dtype='category')
              business_id                          name                    address  ... 4.0 4.5 5.0
0  f9NumwFMBDn751xgFiRbNA      The Range At Lake Norman            10913 Bailey Rd  ...   0   0   0
1  XNoUzKckATkOD1hP6vghZg                       Felinus      3554 Rue Notre-Dame O  ...   0   0   1
2  51M2Kk903DFYI6gnB5I6SQ       USE MY GUY SERVICES LLC         4827 E Downing Cir  ...   0   1   0
3  cKyLV5oWZJ2NudWgqs8VZw   Oasis Auto Center - Gilbert  1720 W Elliot Rd, Ste 105  ...   0   1   0
4  ScYkbYNkDgCneBrD9vqhCQ  Junction Tire & Auto Service        6910 E Southern Ave  ...   0   0   1

[5 rows x 25 columns]
'''

df['webpage_text'].dropna(inplace=True)
df['webpage_text'] = df['webpage_text'].apply(lambda x: preprocess(x))
print(df.head())
'''
              business_id  ...                                       webpage_text
0  f9NumwFMBDn751xgFiRbNA  ...  [shooting, ranges, gun, rental, charlotte, nc,...
1  XNoUzKckATkOD1hP6vghZg  ...                                                NaN
2  51M2Kk903DFYI6gnB5I6SQ  ...  [home, renovations, repairs, phoenix, az, home...
3  cKyLV5oWZJ2NudWgqs8VZw  ...  [home, oasis, auto, centeroasis, auto, center,...
4  ScYkbYNkDgCneBrD9vqhCQ  ...  [contact, junction, tire, tires, auto, repair,...
'''
embedding = word2vec(dim=100)
#print(len(embedding))
#400000



