import statistics 
from cleaner import clean_sentence

STD_CUT = 40

def _count_tokens(corpus):
    mapOfWords = {}
    total = 0
    for data in corpus :
        tokens = data[0]
        for token in tokens:
            if token in mapOfWords:
                mapOfWords[token] += 1
            else:
                mapOfWords[token] = 1
                total += 1
    for k in mapOfWords:
        mapOfWords[k] = mapOfWords[k] / total
    return mapOfWords

def _substract_dictionary(d1, d2):
    toRet = {}
    for k in d1:
        toRet[k] = d1[k] - d2[k]
    return toRet

def _find_cut(values):
    prev = 0
    count = 0
    stdDev = statistics.stdev(values)
    for v in values:
        if abs(prev - v) < stdDev:
            count += 1
        else:
            count = 0
        if count == STD_CUT:
            return v
        prev = v   

class FeatureSelector():
    corpus = []
    selectedTokens = []

    def __init__(self, corpus):
        self.corpus = corpus
        positive = []
        negative = []
        for data in corpus:
            if data[2] is 'positive':
                positive.append(data)
            if data[2] is 'negative':
                negative.append(data)

        baseline = _count_tokens(self.corpus)
        positive = _count_tokens(positive)
        negative = _count_tokens(negative)

        positive = _substract_dictionary(positive, baseline)
        negative = _substract_dictionary(negative, baseline)
        positive = [(k, v, 'positive') for k, v in positive.items()]
        negative = [(k, v, 'negative') for k, v in negative.items()]

        self._select_best_tokens([positive, negative])
    
    def _select_best_tokens(self, listOfTokensCount):
        for tokensCount in listOfTokensCount:
            tokensCount.sort(key=lambda tup: tup[1], reverse = True)
            cut = _find_cut(list(map(lambda x: x[1], tokensCount)))
            for t in tokensCount:
                if(t[1] > cut):
                    self.selectedTokens.append(t[0])
        print("[Tokens soleccionados " + str(len(self.selectedTokens)) + " ]")
        for t in self.selectedTokens:
            print("[     " + t + " ]")
        
    
    def sentence_to_naive_bayes_feature(self, sentence):
        sentenceAsList = clean_sentence(sentence)
        features = dict((el,0) for el in self.selectedTokens)
        for token in sentenceAsList:
            if token in self.selectedTokens:
                features[token] += 1
        return (features, " ".join(sentenceAsList), sentence)

