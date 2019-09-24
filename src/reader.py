# coding=utf-8
import csv
from logger import log, debug

#TODO: por ahora todo en memoria. Despues podemos implementar Streams
def read_files(filenames):
    debug("[Leyendo archivos...]")
    corpus = []
    for filename in filenames :
        with open('../dataset/' + filename + '.csv') as datafile:
            reader = csv.DictReader(datafile)
            for row in reader:
                rate = int(row['RATE'])
                content = row['CONTENT']
                if rate > 3:
                    corpus.append( (content, 'positive') )
                elif rate is 3:
                    corpus.append( (content, 'neutral') )
                else:
                    corpus.append( (content, 'negative') )
            debug("[   " + filename + " leido]")
            
    return corpus