from nltk.corpus import stopwords
from logger import log, debug
import nltk

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
    "UNO",
    "--",
    "PRODUCTO",
    "TAL",
    "MES",
    "$"
]

def clean_sentence(sentence):
    newSentenceAsList = []
    tokens = nltk.word_tokenize(sentence.upper().replace(",", "").replace(".", "").replace("!", ""), 'spanish')
    for token in tokens:
        if token not in fillerWords:
            newSentenceAsList.append(token)
    return newSentenceAsList
        


def clean_corpus(listOfTuples):
    debug("[Limpiando el corpus...]")

    newCorpus = []
    for dataTuple in listOfTuples:
        sentence = dataTuple[0]
        newCorpus.append((clean_sentence(sentence), sentence, dataTuple[1]))
    return newCorpus

