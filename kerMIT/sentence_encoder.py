__author__ = 'lorenzo'

import numpy
from functools import reduce
import operation as op
from tree import Tree
import dtk2

def encoder(sentence, dtk_generator, n=1, mode="pos"):
    if mode == "pos":
        posTags = [x[0] for x in sentence]
        if n == 1:
            return sum(map(dtk_generator.distributedVector, posTags))
        else:
            posTagsByN = list(ngram(posTags, n))
            distributedPosTagsByN = [map(dtk_generator.distributedVector, x) for x in posTagsByN]
            return numpy.sum([reduce(op.fast_shuffled_convolution, x) for x in distributedPosTagsByN]) + encoder(sentence, dtk_generator, n-1, mode)
    else:
        if n == 1:
            v = numpy.zeros(dtk_generator.dimension)
            for pos, word in sentence:
                pos_v = dtk_generator.distributedVector(pos)
                word_v = dtk_generator.distributedVector(word)
                v = v + op.fast_shuffled_convolution(pos_v, word_v)
                return v
        else:
            posTags = [x[0] for x in sentence]
            posTagsByN = list(ngram(posTags, n))
            distributedPosTagsByN = [map(dtk_generator.distributedVector, x) for x in posTagsByN]
            return numpy.sum([reduce(op.fast_shuffled_convolution, x) for x in distributedPosTagsByN]) + encoder(sentence, dtk_generator, n-1, mode)


def ngram(l, n):
    for i in range(len(l) - (n - 1)):
        yield l[i:i+n]

def fileEncoder(path):
    for line in open(path):
        t = Tree(string = line)
        sentence = t.taggedSentence
        v = encoder(sentence, dtk_generator, 3)
        numpy.save(output + "/prova.npy", [v])




if __name__ == "__main__":
    dtk_generator = dtk.DT(dimension=8192, LAMBDA=0.6, operation=op.fast_shuffled_convolution)
    path = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB2/00/wsj_0001.mrgbinarized.txt"
    output = "/Users/lorenzo/Desktop/Current/Greg/dtknn/encoded"
    output = "/home/ferrone/dtknn/encoded"
    for line in open(path):
        t = Tree(string = line)
        sentence = t.taggedSentence
        v = encoder(sentence, dtk_generator, 3)
        numpy.save(output + "/prova.npy", [v])
