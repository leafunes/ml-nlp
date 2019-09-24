from selector import select_best_tokens
from corpus import shuffle, split
from cleaner import clean_corpus
from logger import log, debug


def train(corpus, classifier_trainer, classifier_tester, selector_factory, feature_extractor):
    for i in range(5, 15):
        best_tokens = select_best_tokens(clean_corpus(corpus), selector_factory(i))
        selector = feature_extractor(best_tokens)

        log("[Seleccionando features...]")
        train_corpus, test_corpus = split(shuffle(corpus), 80)

        log("[Clasificando ...]")
        classifier = classifier_trainer(train_corpus, selector)

        log("[Testeando ...]")
        acc = classifier_tester(classifier, test_corpus, selector)

        log("[Porcentaje " + str(acc * 100) + "% ]")
