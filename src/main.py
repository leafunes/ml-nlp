# coding=utf-8
import csv
import re
import matplotlib.pyplot as plt
import statistics 
import nltk
from nltk.corpus import stopwords
from sklearn import svm
import random


fillerWords = stopwords.words('spanish') + [
    "A",
    "ANTE",
    "BAJO",
    "CABE",
    "CON",
    "CONTRA",
    "DE",
    "DESDE",
    "DURANTE",
    "EN",
    "ENTRE",
    "HACIA",
    "HASTA",
    "MEDIANTE",
    "PARA",
    "POR",
    "SEGUN",
    "SIN",
    "SO",
    "SOBRE",
    "TRAS",
    "VERSUS",
    "VIA",
    "EL",
    "LOS",
    "LAS",
    "LA",
    "Y",
    "A",
    "ES",
    "EN",
    "LO",
    "",
    "ME",
    "SE",
    "UN",
    "O",
    "QUE",
    "NO",
    "DEL",
    "YA",
    "SI",
    "Q",
    "AL",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "MUY",
    "UNA",
    "UNO"
    ]


def removeLetters(string):
    return ''.join(sorted(set(string), key=string.index))

def getWords(string):
    newString = string.upper().replace(",", "").replace(".", "").replace("!", "")
    return newString.split(' ') 

def getCountOfWords(filenames):
    mapOfWords = dict()
    total = 0
    for filename in filenames :
        with open('../dataset/' + filename + '.csv') as datafile:
            reader = csv.reader(datafile)
            for row in reader:
                review = row[2]
                for word in getWords(review):
                    if word in fillerWords:
                        continue
                    if word in mapOfWords:
                        mapOfWords[word] += 1
                    else:
                        mapOfWords[word] = 1
                    total += 1
    for k in mapOfWords:
        mapOfWords[k] = mapOfWords[k] / total
    return mapOfWords

def findCut(sortedDict, stdDev):
    prev = 0
    count = 0
    for k, v in sortedDict:
        if abs(prev - v) < stdDev:
            count += 1
        else:
            count = 0
        if count == 40:
            return v
        prev = v        

def clearAndSortDictOfWords(words):
    cleanDict = dict()
    sortedDict = sorted(words.items(), key=lambda kv: kv[1], reverse=True)
    stdDev = statistics.stdev(words.values())

    cut = findCut(sortedDict,stdDev)
    for key, value in sortedDict:
        if(value > cut):
            cleanDict[key] = value
    return cleanDict

def substractDictionary(dict1, dict2):
    toRet = dict()
    for k in dict1:
        toRet[k] = dict1[k] - dict2[k]
    return toRet

def createGraphic(negativeDict, positiveDict):
    plt.rc('xtick', labelsize=8)
    plt.subplots_adjust(hspace = 0.5)

    fig = plt.figure(1, figsize=(200, 600))
    fig.canvas.set_window_title('Reviews de productos de mercadolibre')

    plt.subplot(211, xlabel="Palabras", ylabel="Porporción de aparicion", title="Reviews positivas")
    plt.bar(list(cleanPostive.keys()), cleanPostive.values(), color='g')

    plt.subplot(212, xlabel="Palabras", ylabel="Porporción de aparicion", title="Reviews negativas")
    plt.bar(list(cleanNegative.keys()), cleanNegative.values(), color='r')

    plt.show()

def getNumberByPercentage(percentage, total):
    return int((percentage * total) / 100)

def shuffleAndSplitFiles(filenames):
    corpus = []
    for filename in filenames:
        with open('../dataset/' + filename + '.csv') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row['RATE']) > 3:
                    corpus.append( (row['CONTENT'], True) )
                else:
                    corpus.append( (row['CONTENT'], False) )
    random.shuffle(corpus)
    l = len(corpus)
    percent80 = getNumberByPercentage(80, l) 
    percent15 = getNumberByPercentage(15, l) 
    return corpus[0:percent80], corpus[percent80: percent80 + percent15], corpus[percent80 + percent15: l]

def featureExtractor(text, selectedWords):
    features = {}
    upperText = text.upper()
    for selected in selectedWords:
        features[selected] = upperText.count(selected)
    return features


baseLine = getCountOfWords(['positive', 'negative', 'regular'])
positive = substractDictionary(getCountOfWords(['positive']), baseLine)
negative = substractDictionary(getCountOfWords(['negative']), baseLine)

cleanPostive = clearAndSortDictOfWords(positive)
cleanNegative = clearAndSortDictOfWords(negative)
selectedWords = []
selectedWords += list(cleanPostive.keys())
selectedWords += list(cleanNegative.keys())
 	
train_set, devtest_set, test_set = shuffleAndSplitFiles(['positive', 'negative'])
train_set = list(map(lambda x: (featureExtractor(x[0], selectedWords), x[1]), train_set ))
devtest_set = list(map(lambda x: (featureExtractor(x[0], selectedWords), x[1], x[0]), devtest_set))
test_set = list(map(lambda x: (featureExtractor(x[0], selectedWords), x[1], x[0]), test_set))


svmClassif = svm.SVC(gamma='scale')
x = []
y = []
for e in train_set:
    x.append(list(e[0].values()))
    y.append(e[1])


svmClassif.fit(x, y)

fail = 0
for test in devtest_set:
    predicted = svmClassif.predict([list(test[0].values())])[0]
    real = test[1]
    if bool(predicted) is not bool(real):
        print("falla!! " + str(real) + " " + test[2])
        fail = fail + 1
print(str(fail) + " de " + str(len(devtest_set)))

"""
classifier = nltk.NaiveBayesClassifier.train(train_set)

fail = 0
for test in test_set:
    predicted = classifier.classify(test[0])
    real = test[1]
    if predicted is not real:
        print("falla!! " + str(real) + " " + test[2])
        fail = fail + 1
print(str(fail) + " de " + str(len(test_set)))
print(nltk.classify.accuracy(classifier, devtest_set))
"""



