__author__ = 'lorenzo'


from dtk import DT
from tree import Tree
import numpy as np
from semantic_vector import SemanticVector
import os

class Feature:

    def __init__(self,initializeDT=False, initializeSV=False, sv=None):
        if initializeDT:
            self.k = DT(LAMBDA = 0.4, dimension=1024)

        if (initializeSV) and (sv is not None):
            dir = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/SVM/"
            matrix = "single-target.dm"
            file = os.path.join(dir, matrix)
            self.sv = SemanticVector(sv)
            if initializeDT:
                self.k.sv = self.sv






    def dt(self, pair):
        """concantenazione dei due dt"""
        t1 = Tree(string=pair[0])
        t2 = Tree(string=pair[1])

        dt1 = self.k.dt(t1)
        dt2 = self.k.dt(t2)

        a = np.concatenate((dt1, dt2))

        return a

    def dst(self, pair):
        t1 = Tree(string=pair[0], lexicalized=True)
        t2 = Tree(string=pair[1], lexicalized=True)

        dt1 = self.k.dst(t1)
        dt2 = self.k.dst(t2)

        a = np.concatenate((dt1, dt2))

        return a


    def bow(self, pair):
        w1 = set(w for w in pair[0])
        w2 = set(w for w in pair[1])
        return np.array([len(w1.intersection(w2))])

    def additive(self, elem):

        s1 = sum(self.sv.word(w) for w in Tree(string=elem[0]).sentence())
        s2 = sum(self.sv.word(w) for w in Tree(string=elem[1]).sentence())

        return np.concatenate((s1,s2))
