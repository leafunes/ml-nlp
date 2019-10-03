import nltk

from reader import read_files
from logger import log, debug
import logger
from cleaner import clean_corpus, clean_sentence
from corpus import shuffle, split
from selector import select_best_tokens, sentence_to_naive_bayes_feature, selector_factory_by_stdev, selector_factory_by_label, sentence_to_SVM_feature
from gym import trainer_factory
from sklearn import svm
import numpy as np


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten, SpatialDropout1D
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import load_model
from keras.preprocessing import sequence


logger.DEBUG = False

def sentence_to_best_tokens(selectedTokens):
    def curry(sentenceAsList):
        features = []
        for token in sentenceAsList:
            if token in selectedTokens:
                features.append(token)
        return features
    return curry

def map_sentiment(sentiment):
    if sentiment is 'positive':
        return [1, 0 ,0]
    elif sentiment is 'negative' :
        return [0, 0, 1]
    else:
        return [0, 1, 0]

def map_vector(vector):
    positive = vector[0]
    neutral = vector[1]
    negative = vector[2]
    if(positive > neutral and positive > negative):
        return 'positive'
    if(neutral > positive and neutral > negative):
        return 'neutral'
    if(negative > neutral and negative > positive):
        return 'negative'


def promedio(prediction):
    total = 0
    for e in prediction:
        total += e[0]
    return total / len(prediction)

corpus = read_files(['positive_long', 'negative_long', 'regular_long'])
corpus = clean_corpus(corpus)
corpus = shuffle(corpus)
newCorpus = {}
newCorpus["text"] = []
newCorpus["raw"] = []
newCorpus["sentiment"] = []

max_features = 20000

selector = selector_factory_by_label(max_features)
best_tokens = select_best_tokens(corpus, selector)

feature_extractor = sentence_to_best_tokens(best_tokens)

logger.log("Mapeando corpus")
for (s,ss,l) in corpus:
    newCorpus["text"].append(" ".join(s))
    newCorpus["raw"].append(ss)
    newCorpus["sentiment"].append(map_sentiment(l))

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(newCorpus['text'])
list_tokenized_train = tokenizer.texts_to_sequences(newCorpus['text'])

# cut texts after this number of words (among top max_features most common words)
maxlen = 1500
x_total = pad_sequences(list_tokenized_train, maxlen=maxlen)
y_total = newCorpus['sentiment']

(x_train, x_test) = split(x_total, 80)
(x_train_text, x_test_text) = split(newCorpus['raw'], 80)
(y_train, y_test) = split(y_total, 80)

print(x_train[0])
print(y_train[0])
    
batch_size = 120

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
"""
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
"""


embed_dim = 120
lstm_out = 200

model = Sequential()
model.add(Embedding(max_features, 128, input_length=maxlen))
model.add(Bidirectional(LSTM(64)))
model.add(Dropout(0.5))
model.add(Dense(3, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,  
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print('Test score:', score)
print('Test accuracy:', acc)


model.save('models/all_long_more_words.h5')



model = load_model('models/all_long_more_words.h5')


predicted = model.predict(x_test)
count = 0
for i in range(len(predicted)):
    predicted_lbl = map_vector(predicted[i])
    real_lbl = map_vector(y_test[i])

    if predicted_lbl is not real_lbl:
        count += 1
        print(predicted_lbl + " " + real_lbl + ": " + x_test_text[i])
        print("")
        print("")
print(str(count) + " de " + str(len(predicted)))

"""    
frase_test_org = "Buen joystick, sus puntos positivos son que tiene batería no hay que comprar pilas para recargarla tiene un diseño muy ergonomico, es super compatible con windows apenas se conecta el usb inalambrico y ya se puede empezar a jugar no hace falta instalar ningun otro programa y por su puesto su precio. Sus puntos negativos son que es medio plasticoso las teclas de triangulo cuadrado circulo y x se despintan con el tiempo tengo un aproximado de 3 meses con el producto y se le borraron la x y el circulo y por ultimo los gatillos y el analógico estan muy bien aunque podrían ser mas precisos aunque por el precio que cuesta esta mas que bien y mas si no queremos pagar el doble por uno de xbox o de play."
frase_test = clean_sentence(frase_test_org)
frase_test = tokenizer.texts_to_sequences([frase_test])
x_test = pad_sequences(frase_test, maxlen=maxlen)
"""



print(frase_test_org + " " + map_vector(model.predict(x_test)[0]))

                            

"""
x_train, y_train = np.array(x_train), np.array(y_train)
# Reshaping X_train for efficient modelling
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.2))
# Second LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Third LSTM layer
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
# Fourth LSTM layer
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))
# The output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error')
# Fitting to the training set
regressor.fit(x_train, y_train,epochs=50,batch_size=32)
regressor.save('models/lstm.h5')



regressor = load_model('models/1.h5')
for i in range(len(x_test)):
    print("preditcted: " + str(promedio(regressor.predict(x_test[i]))) + " real: " + str(y_test[i]))

"""