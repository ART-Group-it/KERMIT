__author__ = 'lorenzo'


import os
import numpy as np
import hashlib

dir = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/SVM/"
matrix = "single-target.dm"

class SemanticVector:

    def __init__(self, file, random=True):

        file = open(file)
        self.random = random
        self.dic = {}
        self.dimension = int(file.readline().split(" ")[1])
        for line in file.readlines():
            l = line.split(" ")
            word = l[0]
            self.dic[word] = np.array(l[1:],dtype=float)

    def word(self, word):

        word = word.lower()

        try:
            vett = self.dic[word]
            vett = vett/np.sqrt(sum(vett**2))
            return vett
        except KeyError:
            #print ("word not found")
            if self.random:
                h = int(hashlib.md5(word.encode()).hexdigest(),16) % 1000000000000000
                v = np.random.normal(0,1,self.dimension)
                return v/np.sqrt(sum(v**2))

    def sim(self, a, b):
        w1 = self.word(a)
        w2 = self.word(b)
        return np.dot(w1,w2)




if __name__ == "__main__":

    dir = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/SVM/"
    matrix = "single-target.dm"
    file = os.path.join(dir, matrix)
    sv = SemanticVector(file)

    k = sv.sim("love::n", "hate::n")

    print(k)
