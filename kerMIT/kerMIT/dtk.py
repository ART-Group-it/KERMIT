import numpy as np
import time
import hashlib
import gc

from kerMIT import operation as op
from kerMIT.tree import Tree

#from semantic_vector import SemanticVector


class DT:

    def __init__(self, LAMBDA = 1., dimension=1024, file=None, kernelType="default",operation=op.fast_shuffled_convolution):
        self.LAMBDA = LAMBDA
        self.dimension = dimension
        self.operation = operation
        self.sn_cache = {}
        self.dt_cache = {}
        self.dtf_cache = {}
        self.result = np.zeros(self.dimension)
        if operation in [op.randomOperation, op.randomBilinearOperation]:
            self.mode = "binary"
        else:
            self.mode = "normal"

        if file:
            if isinstance(file,SemanticVector):
                self.sv = file
            else:
                self.sv = SemanticVector(file)

        self.kernelType = kernelType

        self.type_dic = {
            "default" : self.dt,
            "dstk" : self.dst
        }


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


    def cleanCache(self):
        self.sn_cache = {}
        self.dt_cache = {}
        self.dtf_cache = {}
        gc.collect()

    def distributedVector(self, s):
        # h = int(hashlib.md5(s.encode()).hexdigest(),16) % 100000000              #probably too slow and not necessary ??
        # h = abs(mmh3.hash(s)) % 1000000

        h = abs(op.hash(s)) % 4294967295

        # h = np.abs(hash(s))         #devo hashare s in qualche modo (controllare che basti) e
        np.random.seed(h)            #inizializzare np.random.seed()
        if self.mode == "binary":
            q = 1/np.sqrt(self.dimension)
            return np.random.choice([-q, q], self.dimension)
        else:
            return op.random_vector(self.dimension, normalized=False)
        # return np.random.normal(0,1./np.sqrt(self.dimension),self.dimension)

    def sRecursive(self, tree, spectrum):
        result = np.zeros(self.dimension)
        if not tree.isTerminal():
            preterminal = True
            for i,child in enumerate(tree.children):
                childVector = self.distributedVector(child.root)

                childVector = childVector + np.sqrt(self.LAMBDA)*self.sRecursive(child, spectrum)
                #print (child, np.linalg.norm(childVector))
                preterminal = False

                result = childVector if i==0 else self.operation(result, childVector)

            if not preterminal:
                result = self.operation(self.distributedVector(tree.root), result)
                self.result = spectrum + result
        else:
            result = np.zeros(self.dimension)

        return result

    def sn(self, tree):

        #print ("++++++++++")
        if tree in self.sn_cache:


            #pass
            return self.sn_cache[tree]
        if tree.isTerminal():
            self.sn_cache[tree] =  np.zeros(self.dimension)
            return self.sn_cache[tree]
            #return self.distributedVector(tree.root)
        else:
            vec = self.distributedVector(tree.root)
            separator = self.distributedVector("separator")
            vec = self.operation(vec,separator)
            #print ("radice: ", tree.root)
            #vv = functools.reduce(self.operation, [(self.distributedVector(c.root) + self.sn(c)) for c in tree.children])
            #self.sn_cache[tree] = self.operation(vec, vv)
            #return self.sn_cache[tree]
            for c in tree.children:
               # if not c.isTerminal():
                #print ("child: ", c.root)
                vecChildren = np.sqrt(self.LAMBDA)*(self.distributedVector(c.root) + self.sn(c))

                vec = self.operation(vec, vecChildren)

        #print (tree, np.linalg.norm(vec)**2)
        #print ("--------")
        self.sn_cache[tree] = vec
        return self.sn_cache[tree]

    def sn2(self, tree):
        result = np.zeros(self.dimension)

        if not tree.isTerminal():
            preterminal = True
            for i,child in enumerate(tree.children):
                childVector = self.distributedVector(child.root)
                if not child.isTerminal():
                    childVector = childVector + np.sqrt(self.LAMBDA)*self.sn2(child)
                    #print (child, np.linalg.norm(childVector))
                    preterminal = False

                result = childVector if i==0 else self.operation(result, childVector)

            if not preterminal:
                result = self.operation(self.distributedVector(tree.root), result)
            else:
                result = np.zeros(self.dimension)

        return result

    def dt(self, tree):
        # result = np.zeros(self.dimension)
        # self.sRecursive(tree, self.result)
        # return self.result
        if tree in self.dt_cache:
            return self.dt_cache[tree]
        else:
            v = sum(self.sn(node) for node in tree.allNodes())
            self.dt_cache[tree] = v
        return v

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

    def drule(self, tree):
        return sum(self.dt(x) for x in tree.allRules())

    def dst(self, tree):
        """only works on lexicalized trees"""
        if tree.lexicalized:
            v = sum(np.outer(self.sn(node),self.sv.word(node.lemma)) for node in tree.allNodes() if not node.isTerminal())
            v = v.reshape(v.size)
            v = v/np.sqrt(sum(v**2))
            return v
        else:
            raise TypeError("Tree is not lexicalized")

    def kernel(self, t1, t2):
        try:
             f = self.type_dic[self.kernelType]
        except KeyError:
            print("kernelType ", self.kernelType, " not implemented.")

        return f(t1).dot(f(t2))



if __name__ == "__main__":
    #s = "(S#trade$v:-1:4095 (NP#oil$n:-1:63 (NP#oil$n:-1:63 (JJ#Crude$j:1:7 Crude)(NN#oil$n:2:56 oil)))(VP#trade$v:-1:4032 (VBD#trade$v:-1:0 traded)(PP#barrel$n:-1:4032 (IN#at$i:-1:0 at)(NP#barrel$n:-1:4032 (NP#37.80$c:-1:448 ($#$$$:-1:0 $)(CD#37.80$c:3:448 37.80))(NP#barrel$n:-1:3584 (DT#a$d:-1:0 a)(NN#barrel$n:4:3584 barrel))))))"
    ss = "(S (@S (NP (NP (@NP (DT The) (NN wait)) (NN time)) (PP (IN for) (NP (@NP (DT a) (JJ green)) (NN card)))) (VP (AUX has) (VP (@VP (VBN risen) (PP (IN from) (NP (@NP (NP (CD 21) (NNS months)) (TO to)) (NP (CD 33) (NNS months))))) (PP (IN in) (NP (@NP (DT those) (JJ same)) (NNS regions)))))) (. .))"
    ss = '(NOTYPE##ROOT(NOTYPE##NP(NOTYPE##S(NOTYPE##NP(NOTYPE##NNP(NOTYPE##Children)))(NOTYPE##VP-REL(NOTYPE##VBG-REL(NOTYPE##W))(NOTYPE##CC(NOTYPE##and))(NOTYPE##VBG(NOTYPE##waving))(NOTYPE##PP(NOTYPE##IN(NOTYPE##W))(NOTYPE##NP(NOTYPE##NN(NOTYPE##camera))))))))'
    ss = ss.replace(")", ") ").replace("(", " (")

    t = Tree(string=ss)



    kernel = DT(dimension=4000, LAMBDA= 0.6, operation=op.fast_shuffled_convolution)

    v = kernel.drule(t)
    vv = kernel.dt(t)
    for r in t.allRules():
        print (r)
    print (v)
    print (vv)
    print (np.dot(v, vv))


    #print(kernel.kernel(t,frag))
