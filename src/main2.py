import nltk

from reader import read_files
from cleaner import clean_corpus
from corpus import shuffle, split
from selector import FeatureSelector

corpus = read_files(['positive', 'negative', 'regular'])
feature_selector = FeatureSelector(clean_corpus(corpus))

print("[Seleccionando features para NB...]")
train_corpus, test_corpus = split(shuffle(corpus), 80)
train_set = [(feature_selector.sentence_to_naive_bayes_feature(s)[0], l) for (s, l) in train_corpus]
test_set = list(map(lambda x: (feature_selector.sentence_to_naive_bayes_feature(x[0])[0], x[1]), test_corpus))

print("[Clasificando ...]")
classifier = nltk.NaiveBayesClassifier.train(train_set)

print("[Testeando ...]")
acc = nltk.classify.accuracy(classifier, test_set)
print("[Porcentaje " + str(acc * 100) + "% ]")

