# coding=utf-8
import csv
import re
from unicodedata import normalize
import matplotlib.pyplot as plt
import statistics 



fillerWords = [
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
        if count == 3:
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


baseLine = getCountOfWords(['positive', 'negative', 'regular'])
positive = substractDictionary(getCountOfWords(['positive']), baseLine)
negative = substractDictionary(getCountOfWords(['negative']), baseLine)

cleanPostive = clearAndSortDictOfWords(positive)
cleanNegative = clearAndSortDictOfWords(negative)

plt.rc('xtick', labelsize=8)
plt.subplots_adjust(hspace = 0.5)

fig = plt.figure(1, figsize=(200, 600))
fig.canvas.set_window_title('Reviews de productos de mercadolibre')

plt.subplot(211, xlabel="Palabras", ylabel="Porporción de aparicion", title="Reviews positivas")
plt.bar(list(cleanPostive.keys()), cleanPostive.values(), color='g')

plt.subplot(212, xlabel="Palabras", ylabel="Porporción de aparicion", title="Reviews negativas")
plt.bar(list(cleanNegative.keys()), cleanNegative.values(), color='r')

plt.show()
