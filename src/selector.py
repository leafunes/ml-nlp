import statistics 
from cleaner import clean_sentence
from logger import log, debug

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

def _find_cut(values, cutValue):
    prev = 0
    count = 0
    stdDev = statistics.stdev(values)
    for v in values:
        if abs(prev - v) < stdDev:
            count += 1
        else:
            count = 0
        if count == cutValue:
            return v
        prev = v 

def selector_factory_by_stdev(cutValue):
    def factory(listOfTokensCount):
        tokensCount = listOfTokensCount.copy()
        tokensCount.sort(key=lambda tup: tup[1], reverse = True)
        cut = _find_cut(list(map(lambda x: x[1], tokensCount)), cutValue)
        def curry(element):
            return element[1] > cut
        return curry
    return factory

def selector_factory_by_label(cantOfEachLabel):
    def factory(listOfTokensCount):
        tokensCount = listOfTokensCount.copy()
        tokensCount.sort(key=lambda tup: tup[1], reverse = True)
        tokensCount = tokensCount[0: cantOfEachLabel]
        added = 0
        def curry(element):
            return element in tokensCount
        return curry
    return factory

def _calculate_occurences(corpus):
    positive = []
    negative = []
    for data in corpus:
        if data[2] is 'positive':
            positive.append(data)
        if data[2] is 'negative':
            negative.append(data)

    baseline = _count_tokens(corpus)
    positive = _count_tokens(positive)
    negative = _count_tokens(negative)

    positive = _substract_dictionary(positive, baseline)
    negative = _substract_dictionary(negative, baseline)
    positive = [(k, v, 'positive') for k, v in positive.items()]
    negative = [(k, v, 'negative') for k, v in negative.items()]
        
    return [negative, positive]

def select_best_tokens(corpus, selector_factory):
    listOfTokensCount = _calculate_occurences(corpus)
    selectedTokens = []
    for tokensCount in listOfTokensCount:
        selector = selector_factory(tokensCount)
        for t in tokensCount:
            if(selector(t)):
                selectedTokens.append(t[0])
    debug("[Tokens soleccionados " + str(len(selectedTokens)) + " ]")
    for t in selectedTokens:
        debug("[     " + t + " ]")
    
    return selectedTokens

def sentence_to_naive_bayes_feature(selectedTokens):
    def curry(sentence):
        sentenceAsList = clean_sentence(sentence)
        features = dict((el,0) for el in selectedTokens)
        for token in sentenceAsList:
            if token in selectedTokens:
                features[token] += 1
        return (features, " ".join(sentenceAsList), sentence)
    return curry
