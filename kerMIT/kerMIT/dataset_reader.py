__author__ = 'lorenzo'

import os
import numpy as np
import xml.etree.ElementTree as etree
import math
import operation
import tree
import time
import treekernel
from dtk import DT
import scipy.stats
#from feature import Feature

dir = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/RTE"
dev = "RTE3_dev_processed.xml.svm"
test = "RTE3_test_processed.xml"

dict = {"YES":1, "NO":-1}

class Dataset2:
    def __init__(self, file, feature_func=None, processed=False):
        global dict
        f = open(file)
        t = etree.parse(file)
        root = t.getroot()
        self.y = np.array([dict[pair.attrib["entailment"]] for pair in root])
        if processed:
            self.pairs = [(pair[2][1][0].text,pair[2][1][1].text)  for pair in root]
        else:
            self.pairs = [(pair[2][0][0].text,pair[2][0][1].text)  for pair in root]
        self.distances = [float(p[3].text[p[3].text.find("distance")+9:]) for p in root]

        if feature_func:
            self.feature_extraction(feature_func)


    def feature_extraction(self, func):
        vecs = []
        for pair, d in zip(self.pairs, self.distances):
            a = func(pair)
            if not math.isnan(d):
                a = np.append(a,d)
            else:
                a = np.append(a,0)
            vecs.append(a)

        self.X = np.array(vecs)




class Dataset:
    def __init__(self, file, feature_func=None):
        f = open(file)
        scores = []
        pairs = []
        distances = []

        for line in f.readlines():
            l = line.split("|BT| ")
            distances.append(float(l[-1].split("|BV|")[-1][4:-6]))
            scores.append(int(l[0].split("\t")[0]))
            pairs.append((l[1], l[2]))
            #dataset.append(((l[1], l[2]),score))

        f.close()
        self.pairs = pairs
        self.y = np.array(scores)
        self.distances = distances

        if feature_func:
            self.feature_extraction(feature_func)

    def feature_extraction(self, func):
        vecs = [func(pair) for pair in self.pairs]     #sta a func() ritornare numpy array
        z = list(zip(vecs, self.distances))
        self.X = np.array(z)




def func(element):
    """from element to vector representation"""
    a, b = element
    return [len(a), len(b)]





if __name__ == "__main__":
    s = os.path.join(dir, test)
    #s = open(s)
    #t = etree.parse(s)
    #root = t.getroot()
    #print(root[0][2][1][0].text)


    D = Dataset2(s, func)


    LAMBDA = 0.4
    dt1 = DT(dimension= 4096, LAMBDA=LAMBDA, operation=operation.random_op)
    dt2 = DT(dimension=4096, LAMBDA=LAMBDA)
    tk = treekernel.TreeKernel(LAMBDA=LAMBDA)

    lista_conv = []
    lista_tk = []
    lista_rand = []

    # start_rand = time.time()


    s1 = (D.pairs[0][0])
    s2 = (D.pairs[0][1])




    #s2 = "(NP (NNP Final) (NNP Fantasy) (NNP III))"

    #s2 = "(A (B c) (B dimension) (B e))"


    print (s1)
    print (s2)
    t1 = tree.Tree(string=s1)
    t2 = tree.Tree(string=s2)

    print (t1.sentence.split(), len(t1.sentence.split()))
    print (t2.sentence.split(), len(t2.sentence.split()))

    # for tt in t2.allNodes():
    #     print (tt, tt.isPreTerminal(), tk.evaluate(tt, tt), dt2.kernel(tt, tt), np.dot(dt2.sn2(tt), dt2.sn2(tt)))

    # print (dt2.kernel(t1, t2))
    # print (tk.evaluate(t1, t2))
    #
    #
    # #



   #CICLO

    for s in D.pairs:
        l1 = [tree.Tree(string=i) for i in s ]

        a = dt1.kernel(*l1)
        b = dt2.kernel(*l1)
        c = tk.evaluate(*l1)

        lista_rand.append(a)
        lista_conv.append(b)
        lista_tk.append(c)





    print ("conv: ", scipy.stats.spearmanr(lista_conv, lista_tk))
    print ("rand: ", scipy.stats.spearmanr(lista_rand, lista_tk))
    #
    #
    #
    #
    #
    #
    #
    #
