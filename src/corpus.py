import random
import numpy

def shuffle(corpus):
    newCorpus = corpus.copy()
    random.shuffle(newCorpus)
    return newCorpus

def _get_percentage(percentage, total):
    return int((percentage * total) / 100)

def equal_split(corpus, parts):
    return numpy.array_split(numpy.array(corpus),parts)

def split(corpus, percentage):
    l = len(corpus)
    percentage = _get_percentage(percentage, l) 
    return corpus[0:percentage], corpus[percentage: percentage + l]