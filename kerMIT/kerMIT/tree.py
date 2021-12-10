'''
Created on Jul 10, 2014

@author: lorenzo
'''

import copy
import numpy as np


class Tree:
    global_id = 0
    def __init__(self, root = None, children = None, string = None, lexicalized=False, id=None, parent=None):
        if id==None: self._id = Tree.global_id
        else: self._id = id
        Tree.global_id += 1

        self.parent = parent

        if string is None:
            self.root = root
            self.children = children

        else:
            try:
                self.parse(string)
                self.cleanup()
            except:
                raise Exception('Insert a valid Tree string')

        if lexicalized:
            self.lexicalized()

        self.string = string
        self.sentence = self.sentence_()
        self.lexicalized = lexicalized
        self.taggedSentence = self.sentence_(posTag=True)
        self.activation_level_initialized = False
        self.activation_level = None
        self.wasTerminal = False # True if this node was a terminal node in the original tree


    def cleanup(self):
        #t = Tree()
        try:
            for i in self.allNodes():
                i.root = i.root.strip(")")
                if not i.isTerminal():
                    if i.children[0] == "":
                        i.children = None

            return self

        except:
            raise Exception('Cleanup Error')

    def label(self):
        return self.root

    def set_activations(self,activation_value):
        self.activation_level = activation_value

    def update_activations(self,activation_value):
        if not self.activation_level_initialized :
            self.set_activations(activation_value)
            self.activation_level_initialized = True
        else:
            self.activation_level = np.add(self.activation_level,activation_value)


    def activation(self):
        return self.activation_level

    def id(self):
        return self._id

    def set_id(self,id):
        self._id = id

    @staticmethod
    def fromstring(tree):
        return Tree(string=tree)


    @staticmethod
    def biggestPars(s):
        count = 0
        pars = []
        for position, char in enumerate(s):
            if char == "(":
                count = count + 1
                if count == 1:
                    startPos = position
            if char == ")":
                count = count - 1
                if count == 0 and position != 0:
                    endPos = position
                    pars.append(s[startPos:endPos+1])

        return pars

    def parse(self, string):
        p = Tree.biggestPars(string)

        if len(p) == 1:
            if p[0].count("(") == 1 and p[0].count(")") == 1:
                root, _, child = p[0].partition(" ")
                self.root = root[1:]
                self.children = [Tree(root = child[:-1], parent=self._id)]

            else:
                root, _, rest = p[0].partition(" ")
                children = []
                self.root = root[1:]

                for c in Tree.biggestPars(rest):
                    #nt = parse(c)
                    nt = Tree(string=c, parent=self._id)
                    children.append(nt)
                self.children = children

    def __eq__(self, other):
        return self.__str__() == other.__str__()

    def __hash__(self):
        return hash(self.__str__())

    def __str__(self):
        if self.isTerminal():
            #print (self.root.left)
            return self.root
        if self.isPreTerminal():
            #return "(" + self.root + " " + self.children[0].root + ")"
            return "(" + self.root + " " + " ".join(c.__str__() for c in self.children) + ")"
            #return self.root + " (" + " ".join(c.__helper_str__() for c in self.children) + ")"

        else:
            return "(" + self.root + " " + " ".join(c.__str__() for c in self.children) + ")"
            #return self.root + " (" + " ".join(c.__helper_str__() for c in self.children) + ")"

    def __repr__(self):
        return self.__str__()
    #def __str__(self):
    #    return "(" + self.__helper_str__() + ")"


    # def __str__(self):
    #     if self.isTerminal():
    #         return self.root
    #     if self.isPreTerminal():
    #         return self.root + " -> " + " , ".join(str(c) for c in self.children)
    #
    #     else:
    #         return self.root + " -> {" +  " , ".join(str(c) for c in self.children) + "}"

    def __len__(self):
        if self.children != None:
            return len(self.children)
        else:
            return 0

    def __getitem__(self, item):
        #if self.children != None :
        return self.children[item]
        #else:
        #    return None

    def __index__(self,i):
        return self.children[i]

    def isTerminal(self):
        return self.children is None

    def isPreTerminal(self):
        if self.children is None:
            return False
        else:
            return all(c.isTerminal() for c in self.children)

    def hasSingleProduction(self):
        for n in self.allNodes():
            if not (n.isPreTerminal() or n.isTerminal()):
                if len(n.children) == 1:
                    return True
        return False

    def singleNode(self):
        if not (self.isPreTerminal() or self.isTerminal()):
            if len(self.children) == 1:
                return True
        return False

#     def binarize(self):
#         if not self.hasSingleProduction():
#             return self
#         for n in self.allNodes():
#             if n.singleNode():
#                 subtrees = n.children[0].children
#                 n.children = subtrees
#         return self

    def binarize(self):
        while self.singleNode():
            subtrees = self.children[0].children
            self.children = subtrees
        if not self.isPreTerminal():
            for sub in self.children:
                sub.binarize()
        return self

    #SBAGLIATA
    def debinarize(self):
        if not self.isPreTerminal():
            if self.root.startswith("@"):
                return self.debinarize_list(self.children)
            else:
                self.children = self.debinarize_list(self.children)
        return [self]

    def debinarize_list(self,list_of_trees):
        new_list = []
        for t in list_of_trees:
            new_list = new_list + t.debinarize()
        return new_list

    def normalize(self):
        for n in self.allTerminalNodes():
            n.root = n.root.lower()
        return self


    def allNodes(self):
        yield self
        if not self.isTerminal():
            for c in self.children:
                yield from c.allNodes()

    def leaves(self):
        return self.allTerminalNodes()

    def allTerminalNodes(self):
        for n in self.allNodes():
            if n.isTerminal():
                yield n
            else:
                continue

    def topRule(self):
        if not self.isTerminal():
            children = [Tree(x.root,None) for x in self.children]
            return Tree(self.root, children)


    def allRules(self):
        for node in self.allNodes():
            if not node.isTerminal():
                children = [Tree(x.root,None) for x in node.children]
                t = Tree(node.root, children)
                t.terminalRule = node.isPreTerminal()
                yield t

    def sentence_(self, posTag=False):
        # if False:
        #     #print (123)
        #     return " ".join([n.root for n in self.allNodes() if n.isTerminal()])
        # else:
        if posTag:
            l = [(n.root, n.children[0].root) for n in self.allNodes() if n.isPreTerminal()]
            return l
        else:
            l = [n.root for n in self.allNodes() if n.isTerminal()]
            if l != [None]:
                return " ".join(l)
            else:
                return None

    def span(self):
        l = [n.root for n in self.allNodes() if n.isTerminal()]
        if l != [None]:
            return l
        else:
            return None



    def depth(self):
        if self.isTerminal():
            return 1
        else:

            return (1 + max(c.depth() for c in self.children))

    def lexicalized(self):
        for n in self.allNodes():
            if n.isTerminal():
                n.lemma = n.root
            else:
                s = n.root.split("#")
                n.root = s[0]
                n.lemma = s[1].split(":")[0].replace("$", "::")

    def add(self, child, position):
        tt = copy.deepcopy(self)
        for i, n in enumerate(tt.allTerminalNodes()):
            if position == i:
                #magari controllare che le root coincidano
                n.children = child.children

        return tt


    def removeWords(self):
        tt = copy.deepcopy(self)
        for n in tt.allNodes():
            if n.isPreTerminal():
                n.children = None

        return tt


    def minmaxactivations(self):
        minmax=(self.activation_level[0],self.activation_level[0])
        for t in self.allNodes():
            for a in t.activation_level:
                if a < minmax[0]:
                    minmax=(a,minmax[1])
                if a > minmax[1]:
                    minmax=(minmax[0],a)
        return minmax

    def minmaxactivations_per_class(self):
        minmax=(self.activation_level.copy(),self.activation_level.copy())
        for t in self.allNodes():
            for i in range(0,len(t.activation_level)):
                if t.activation_level[i] < minmax[0][i]:
                    minmax[0][i] = t.activation_level[i]
                if t.activation_level[i] > minmax[1][i]:
                    minmax[1][i] = t.activation_level[i]
        return minmax

    # restituisce un dizionario in cui ogni nodo id è associato al corrispettivo nodo root
    def getIdNodeAssociation(self):
        d = {}
        for n in self.allNodes():
            d[n._id] = n.root
        return d

    # restituisce un dizionario in cui la chiave è un nodo e il valore è il padre
    def getDictChildParent(self):
        d = {}
        for n in self.allNodes():
            d[n._id] = n.parent
        return d

    # funzione che restituisce la lista di nodi necessare ad arrivare da se stesso alla root dell'albero il path
    # bisogna passare come parametro il dizionario figlio-padre
    def getPathFromALeaveToRoot(self, dict_parent, dict_index):
        start = self._id
        path = []
        while start != None:
            path.append(dict_index[ start ])
            start = dict_parent[start]
        return path

    # prende una matrice in cui ogni riga è una lista di percorsi foglia-root (es: [il, DT, NP])
    # viene restituita una lista in cui in ogni cella c'è il percorso in forma alborea (es: (NP (DT il)) )
    def matrixToString(self,matrix_paths):
        matrix_tree = []
        for row in matrix_paths:
            s = '(' + row[1] + ' ' + row[0] + ')'
            for r in row[2:]:
                s = '(' + r + ' ' + s + ')'
            matrix_tree.append(s)
        return matrix_tree

    # dato un oggetto Tree, restituisce tutti i percorsi dalle foglie del tree alla root
    def leavesToRoot(self):
        dict_index = self.getIdNodeAssociation()
        dict_parent = self.getDictChildParent()
        print(dict_index,dict_parent)
        matrix_paths = [l.getPathFromALeaveToRoot(dict_parent, dict_index) for l in self.leaves()]
        print(matrix_paths)
        matrix_paths = self.matrixToString(matrix_paths)
        return matrix_paths


if __name__ == "__main__":
    treeString4 = "(S (@S (@S (@S (INTJ no) (, ,)) (NP it)) (VP (@VP (VBD was) (RB n't)) (NP (NNP black) (NNP monday)))) (. .))"
    t = Tree(string=treeString4)
    tt = t.removeWords()
    print (tt)
    print ('---')
    for n in tt.allNodes():
        print (n.root, n.children)
    treeString4 = '(S(SBARQ    (WHNP (WP What))    (SQ (VBZ is)      (NP        (NP (DT the) (JJ full) (NN form))        (PP (IN of))))    (. .))(FRAG    (NP (NN com))    (. ?)))'
    print('---\n',treeString4)
    t = Tree(string=treeString4)
    list_of_path = t.leavesToRoot()
    print(list_of_path)