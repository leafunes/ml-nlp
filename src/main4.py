from reader import read_files
import logger
from cleaner import clean_corpus
from corpus import shuffle, split
import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , LSTM , Embedding, Dropout, Bidirectional, Conv1D
from keras.models import Model, Sequential
from keras.models import load_model
from keras.optimizers import Adam

def sentiment_to_vector(sentiment):
    if sentiment is 'positive':
        return [1, 0 ,0]
    elif sentiment is 'negative' :
        return [0, 0, 1]
    else:
        return [0, 1, 0]

def vector_to_sentiment(vector):
    #TODO: esta mal
    positive = vector[0]
    neutral = vector[1]
    negative = vector[2]
    if(positive > neutral and positive > negative):
        return 'positive'
    if(neutral > positive and neutral > negative):
        return 'neutral'
    if(negative > neutral and negative > positive):
        return 'negative'

logger.DEBUG = True


#Hyperparameters
max_features = 2048
maxlen = 128
batch_size = 128
split_percentage = 80
epoch = 7
embed_dim = 128
lstm_out_space = 128
filters = 64
kernel_size = 3

#Parsing
logger.debug("[Abriendo corpus... ]")

corpus = read_files(['positive', 'negative', 'regular'])
corpus = clean_corpus(corpus)
corpus = shuffle(corpus)
newCorpus = {
    'text': [],
    'raw': [],
    'sentiment': []
}

logger.debug("[Mapeando corpus... ]")
logger.debug("[El corpus tiene " + str(len(corpus)) + " rows]")
for (processed_sentence, raw_sentence, sentiment) in corpus:
    newCorpus["text"].append(" ".join(processed_sentence))
    newCorpus["raw"].append(raw_sentence)
    newCorpus["sentiment"].append(sentiment_to_vector(sentiment))

logger.debug("[Vectorizando corpus... ]")
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(newCorpus['text'])
vectorized_text = tokenizer.texts_to_sequences(newCorpus['text'])
logger.log("[Son " + str(len(vectorized_text)) + " total sequences]")


x_total = pad_sequences(vectorized_text, maxlen=maxlen)
y_total = newCorpus['sentiment']

(x_train, x_test) = split(x_total, split_percentage)
(y_train, y_test) = split(y_total, split_percentage)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

logger.log("[Son " + str(len(x_train)) + " train sequences]")
logger.log("[Son " + str(len(x_test)) + " test sequences]")

x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

#Modelo
#TODO: extraer hiperparametros
logger.log("[Buildeando modelo... ]")

"""
model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))
"""
"""
model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=maxlen))
model.add(Dropout(0.1))
model.add(Bidirectional(LSTM(lstm_out_space)))
model.add(Dropout(0.1))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
"""

model = Sequential()
model.add(Embedding(max_features, embed_dim))
model.add(Bidirectional(LSTM(128,  return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=Adam(1e-4),
              metrics=['accuracy'])

logger.log("[Fiteando modelo... ]")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epoch,  
          validation_data=(x_test, y_test))

logger.log("[Testeando modelo... ]")
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

logger.log("[   Score: " + str(score))
logger.log("[   Accuaracy: " + str(acc))
logger.debug("[Guardando modelo... ]")
model.save("models/all_model_2.h5")