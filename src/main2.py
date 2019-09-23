import nltk

from reader import read_files
from cleaner import clean_corpus
from corpus import shuffle, split
from selector import select_best_tokens, sentence_to_naive_bayes_feature, selector_factory_by_stdev

corpus = read_files(['positive', 'negative', 'regular'])
best_tokens = select_best_tokens(clean_corpus(corpus), selector_factory_by_stdev(10))
selector = sentence_to_naive_bayes_feature(best_tokens)

print("[Seleccionando features para NB...]")
train_corpus, test_corpus = split(shuffle(corpus), 80)
train_set = [(selector(s)[0], l) for (s, l) in train_corpus]
test_set = list(map(lambda x: (selector(x[0])[0], x[1]), test_corpus))

print("[Clasificando ...]")
classifier = nltk.NaiveBayesClassifier.train(train_set)

print("[Testeando ...]")
acc = nltk.classify.accuracy(classifier, test_set)
print("[Porcentaje " + str(acc * 100) + "% ]")

