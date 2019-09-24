import nltk

from reader import read_files
from logger import log, debug
import logger
from cleaner import clean_corpus
from corpus import shuffle, split
from selector import select_best_tokens, sentence_to_naive_bayes_feature, selector_factory_by_stdev, selector_factory_by_label
from gym import train


logger.DEBUG = False

corpus = read_files(['positive', 'negative', 'regular'])

def NB_classifier_trainer(train_corpus, selector):
    train_set = [(selector(s)[0], l) for (s, l) in train_corpus]
    return nltk.NaiveBayesClassifier.train(train_set)

def NB_classifier_tester(classifier, test_corpus, selector):
    test_set = list(map(lambda x: (selector(x[0])[0], x[1]), test_corpus))
    return nltk.classify.accuracy(classifier, test_set)

train(corpus, NB_classifier_trainer, NB_classifier_tester, selector_factory_by_label, sentence_to_naive_bayes_feature)

