# coding=utf-8
import csv

#TODO: por ahora todo en memoria. Despues podemos implementar Streams
def read_files(filenames):
    print("[Leyendo archivos...]")
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
            print("[   " + filename + " leido]")
            
    return corpus