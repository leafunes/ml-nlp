import nltk

from reader import read_files
from logger import log, debug
import logger
from cleaner import clean_corpus
from corpus import shuffle, split
from selector import select_best_tokens, sentence_to_naive_bayes_feature, selector_factory_by_stdev, selector_factory_by_label, sentence_to_SVM_feature
from gym import trainer_factory
from sklearn import svm


logger.DEBUG = False

corpus = read_files(['positive', 'negative', 'regular'])

def NB_classifier_trainer(train_corpus, selector):
    train_set = [(selector(s)[0], l) for (s, l) in train_corpus]
    return nltk.NaiveBayesClassifier.train(train_set)

def NB_classifier_tester(classifier, test_corpus, selector):
    test_set = list(map(lambda x: (selector(x[0])[0], x[1]), test_corpus))
    return nltk.classify.accuracy(classifier, test_set)

def SVM_classifier_trainer(train_corpus, selector):
    svmClassif = svm.SVC(gamma='scale')

    x = [selector(s)[0] for (s, l) in train_corpus]
    y = [l for (s, l) in train_corpus]

    svmClassif.fit(x, y)
    return svmClassif

def SVM_classifier_tester(classifier, test_corpus, selector):
    x = [selector(s)[0] for (s, l) in test_corpus]

    predicted = classifier.predict(x)
    hits = 0
    for i in len(predicted):
        if bool(predicted[i]) is bool(test_corpus[1][i]):
            hits += 1
    
    return hits * len(predicted)

def get_max_len():
    pass

train = trainer_factory(3, 20, 10, 100)

train(corpus, NB_classifier_trainer, NB_classifier_tester, selector_factory_by_label, sentence_to_naive_bayes_feature)

#best_tokens = select_best_tokens(cleared_corpus, selector_factory(i))

#max_len = sorted(list(map(lambda x : select_best_tokens(x[0], clean_corpus(corpus))), key= lambda y: len(y), reverse=True)[0]

#log(len(max_len))

#train(corpus, SVM_classifier_trainer, SVM_classifier_tester, selector_factory_by_label, sentence_to_SVM_feature(max_len))

