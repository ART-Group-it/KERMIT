__author__ = 'lorenzo'

import os
import numpy as np
import dataset_reader as dr
from sklearn import svm
from sklearn import tree
from dtk import DT
from tree import Tree
from semantic_vector import SemanticVector
import math
from feature import Feature


# class Learner:
#     def __init__(self, dev, test, func):
#         self.func = func
#         self.dev = dr.Dataset(dev, func)
#         self.test = dr.Dataset(test, func)
#
#     def fit(self):
#         clf = svm.SVC()
#         clf.fit(self.dev.X, self.dev.y)
#
#     @staticmethod
#     def poly_kernel(a,b):
#         p = (1 + np.dot(a[:,-1].reshape((len(a),1)),b[:,-1].reshape((len(b),1)).T))**2
#         #print(p.shape)
#         return p


if __name__ == "__main__":
    dir = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/RTE"
    dev = "RTE3_dev_processed.xml"
    test = "RTE3_test_processed.xml"

    dir_ = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/SVM/"
    matrix = "single-target.dm"
    sm = os.path.join(dir_, matrix)

    #sv = SemanticVector(sm, True)


    def poly_kernel(a,b):
        p = (1 + np.dot(a[:,-1].reshape((len(a),1)),b[:,-1].reshape((len(b),1)).T))**2
        #print(p.shape)

        return p

    def my_kernel(a, b):

        p = np.dot(a[:,:-1], b[:,:-1].T)
        #print(p.shape)

        return p + poly_kernel(a,b)





    F = Feature(True, True, sm)
    dev = dr.Dataset2(os.path.join(dir, dev), F.dst, processed=True)
    test = dr.Dataset2(os.path.join(dir, test), F.dst, processed=True)

    X = dev.X
    y = dev.y


    clf = svm.SVC(C = 0.3, kernel=my_kernel)
    clf.fit(X,y)

    # clf = tree.DecisionTreeClassifier()
    # clf = clf.fit(X,y)

    results = clf.predict(test.X)
    #print(results)
    mistakes = sum(results != test.y)
    print(((800 - mistakes)/800) * 100)
