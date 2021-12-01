__author__ = 'znz8'
import numpy as np
import time
import hashlib
import gc

from kerMIT.tree import Tree
#import operation as op
from kerMIT import operation as op
#from semantic_vector import SemanticVector


class DT:
    #
    def __init__(self, LAMBDA = 1., dimension=4096, operation=op.fast_shuffled_convolution):
        self.LAMBDA = LAMBDA
        self.dimension = dimension
        self.operation = operation
        self.sn_cache = {}
        self.dt_cache = {}
        self.dtf_cache = {}
        self.random_cache = {}
        self.result = np.zeros(self.dimension)
        self.spectrum = np.zeros(self.dimension)


    def cleanCache(self):
        self.sn_cache = {}
        self.dt_cache = {}
        self.dtf_cache = {}
        self.random_cache = {}
        gc.collect()

    def distributedVector(self, s):
        if s in self.random_cache:
            return self.random_cache[s]
        # h = int(hashlib.md5(s.encode()).hexdigest(),16) % 100000000              #probably too slow and not necessary ??
        # h = abs(mmh3.hash(s)) % 1000000

        h = abs(op.hash(s)) % 4294967295

        # h = np.abs(hash(s))         #devo hashare s in qualche modo (controllare che basti) e
        np.random.seed(h)            #inizializzare np.random.seed()
        v_ = op.random_vector(self.dimension,normalized=False)
        self.random_cache[s] = v_
        return v_
        # return np.random.normal(0,1./np.sqrt(self.dimension),self.dimension)

    def sRecursive(self, tree):
        result = np.zeros(self.dimension)
        if tree in self.sn_cache:
            return self.sn_cache[tree]
        if not tree.isTerminal():

            rootVector = self.distributedVector(tree.root)
            separator = self.distributedVector("separator")
            result = self.operation(rootVector, separator)

            for child in tree.children:

                vecChildren = np.sqrt(self.LAMBDA)*(self.distributedVector(child.root) + self.sRecursive(child))
                result = self.operation(result, vecChildren)

        self.spectrum = self.spectrum + result

        return result


    def dt(self, tree):
        self.spectrum = np.zeros(self.dimension)
        self.sRecursive(tree)
        return self.spectrum

    @staticmethod
    def subtrees(tree):
        setOfTrees = set()
        setOfSubTrees = set()
        if tree.isPreTerminal():
            setOfTrees.add(tree)
            setOfSubTrees.add(tree)
        else:
            baseChildrenList = [[]]
            for child in tree.children:
                newBaseChildrenList = []
                c_setOfTrees, c_setOfSubtrees = DT.subtrees(child)
                setOfSubTrees = setOfSubTrees.union(c_setOfSubtrees)
                for treeSubChild in c_setOfTrees:
                   for treeSub in baseChildrenList:
                       newBaseChildrenList.append(treeSub + [treeSubChild])
                baseChildrenList = newBaseChildrenList
            for children in baseChildrenList:
                newTree = Tree(root=tree.root,children=children, id=tree.id())
                setOfTrees.add(newTree)
                setOfSubTrees.add(newTree)

        setOfTrees.add(Tree(root=tree.root, id=tree.id()))
        return setOfTrees,setOfSubTrees

    def dtf(self, tree):
        if tree in self.dtf_cache:
            return self.dtf_cache[tree]
        if tree.isTerminal():
            self.dtf_cache[tree] = self.distributedVector(tree.root)
            return self.dtf_cache[tree]

        else:
            vec = self.distributedVector(tree.root)
            separator = self.distributedVector("separator")
            vec = self.operation(vec,separator)
            for c in tree.children:
               # if not c.isTerminal():
                #print ("child: ", c.root)
                vecChildren = self.dtf(c)
                vec = self.operation(vec, vecChildren)
                #vec = self.operation(vec, dtf(c))

        #print (tree, np.linalg.norm(vec)**2)
        #print ("--------")
        self.dtf_cache[tree] = vec
        return self.dtf_cache[tree]

class partialTreeKernel(DT):
    def __init__(self, LAMBDA = 1., MU = 1., dimension=4096, operation=op.fast_shuffled_convolution):
        super(partialTreeKernel, self).__init__(LAMBDA = LAMBDA, dimension = dimension, operation = operation)
        print ('using partial')
        self.result = np.zeros(self.dimension)
        self.spectrum = np.zeros(self.dimension)
        self.dvalues = {}
        self.mu = MU
        self.MAXPOWER = 10
        self.mus = [self.mu ** i for i in range(self.MAXPOWER)]

    def sRecursive(self, node):
        v = self.distributedVector(node.root)
        result = np.zeros(self.dimension)
        if node.isTerminal():
            result = self.mu * v
        else:
            result = self.operation(self.mu*v, self.d(node.children))

        result =  np.sqrt(self.LAMBDA) * result
        self.spectrum = self.spectrum + result
        return result

    def d(self, trees):

        result = self.dRecursive(trees, 0)
        for i, c in enumerate(trees):
            result = result + self.dRecursive(trees, i)

        return result

    def dRecursive(self, trees, i):
        if i in self.dvalues:
            return self.dvalues[i]

        sci = self.sRecursive(trees[i])
        result = np.zeros(self.dimension)
        if i < len(trees) - 1:
            total = self.dRecursive(trees, i+1)
            for k in range(i+2, len(trees)):
                total = total + self.mus[k - 1 - 1] * self.dRecursive(trees, k)

        else:
            result = sci
        self.dvalues[i] = result
        return result

    def dt(self, tree):
        self.dvalues = {}
        self.spectrum = np.zeros(self.dimension)
        self.sRecursive(tree)
        return self.spectrum








if __name__ == "__main__":
    #s = "(S#trade$v:-1:4095 (NP#oil$n:-1:63 (NP#oil$n:-1:63 (JJ#Crude$j:1:7 Crude)(NN#oil$n:2:56 oil)))(VP#trade$v:-1:4032 (VBD#trade$v:-1:0 traded)(PP#barrel$n:-1:4032 (IN#at$i:-1:0 at)(NP#barrel$n:-1:4032 (NP#37.80$c:-1:448 ($#$$$:-1:0 $)(CD#37.80$c:3:448 37.80))(NP#barrel$n:-1:3584 (DT#a$d:-1:0 a)(NN#barrel$n:4:3584 barrel))))))"
    ss = "(S (@S (NP (NP (@NP (DT The) (NN wait)) (NN time)) (PP (IN for) (NP (@NP (DT a) (JJ green)) (NN card)))) (VP (AUX has) (VP (@VP (VBN risen) (PP (IN from) (NP (@NP (NP (CD 21) (NNS months)) (TO to)) (NP (CD 33) (NNS months))))) (PP (IN in) (NP (@NP (DT those) (JJ same)) (NNS regions)))))) (. .))"
    ss = '(NOTYPE##ROOT(NOTYPE##NP(NOTYPE##S(NOTYPE##NP(NOTYPE##NNP(NOTYPE##Children)))(NOTYPE##VP-REL(NOTYPE##VBG-REL(NOTYPE##W))(NOTYPE##CC(NOTYPE##and))(NOTYPE##VBG(NOTYPE##waving))(NOTYPE##PP(NOTYPE##IN(NOTYPE##W))(NOTYPE##NP(NOTYPE##NN(NOTYPE##camera))))))))'
    ss = ss.replace(")", ") ").replace("(", " (")

    t = Tree(string=ss)



    kernel = partialTreeKernel(dimension=8192, LAMBDA= 0.6, operation=op.fast_shuffled_convolution)

    v = kernel.sRecursive(t)
    w = kernel.dt(t)

    print (v)
    print (w)



    #print(kernel.kernel(t,frag))
