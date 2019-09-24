from selector import select_best_tokens
from corpus import shuffle, split, equal_split
from cleaner import clean_corpus
from logger import log, debug


def trainer_factory(min, max, break_threshold, iterations):
    def with_range(corpus, classifier_trainer, classifier_tester, selector_factory, feature_extractor):
        cleared_corpus = clean_corpus(corpus)

        log("[Seleccionando features...]")
        train_corpus, test_corpus = split(shuffle(corpus), 80)

        for i in range(min, max):
            best_tokens = select_best_tokens(cleared_corpus, selector_factory(i))
            selector = feature_extractor(best_tokens)
            
            log("[Entrenando con parametro " + str(i) + "]")
            log("------------------------")
            
            best_classifier = _super_train_model(train_corpus, classifier_trainer, classifier_tester, selector, break_threshold, iterations)

            acc = classifier_tester(best_classifier, test_corpus, selector)
            log("[Resultado  " + str(acc * 100) + "% ]")
    return with_range



def _super_train_model(train_set, classifier_trainer, classifier_tester, selector, break_threshold, iterations):
    train_set_splitted = equal_split(train_set, 5)
    max_classifier = (None, 0)

    splitted_len = len(train_set_splitted)
    for i in range(splitted_len):
        no_better_count = 0
        for _ in range(iterations):
            if(no_better_count > break_threshold):
                break
            train_subset = train_set_splitted[0:i] + train_set_splitted[(i + 1): splitted_len]
            train_subset = [item for sublist in train_subset for item in sublist]
            train_subset = shuffle(train_subset)

            test_subset = shuffle(train_set_splitted[i])
                
            classifier = classifier_trainer(train_subset, selector)
            acc = classifier_tester(classifier, test_subset, selector)
            if(acc > max_classifier[1]):
                log("[Se encontro un clasificador mejor " + str(acc * 100) + "% ]")
                max_classifier = (classifier, acc)
            no_better_count += 1

    return max_classifier[0]

