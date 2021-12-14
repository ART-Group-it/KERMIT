import numpy as np
from kerMIT import operation as op
from kerMIT.tree import Tree
from kerMIT.dtk import DT


class ActivationSubtreeLRP:
    def __init__(self, kernel):
        self.kernel = kernel


    def on_demand_embedding_matrix(self, tree):
        '''
        From the kernel I extract all the subtrees and generate 4 dictionaries
        :param kernel:
        :return: tree_index: {subtree: vector}
        :return: tree_index_dict: {subtree: index}
        :return: embedding_matrix: Ordered vector matrix np.array(np.array())
        :return: tree_index_dict_inverse: {index: subtree}
        '''
        tree_index = {}
        tree_index_dict = {}
        tree_index_dict_inverse = {}
        embedding_matrix = []
        weights = []
        count = 0
        for k in DT.subtrees(tree)[1]:
            (dd, w) = self.kernel.dtf_and_weight(k)
            tree_index[k] = dd
            tree_index_dict[k] = count
            embedding_matrix.append(dd)
            tree_index_dict_inverse[count] = k
            weights.add(w)
            count += 1
        self.kernel.cleanCache()
        return tree_index, tree_index_dict, np.array(embedding_matrix), tree_index_dict_inverse, np.arrary(weights)

    def calculateDTFs(self, tree):
        '''
        Takes a tree as a string and generates the dtf, so I save it in cache
        :param tree: str
        :return: 
        '''
        for tt in DT.subtrees(tree)[1]:
           self.kernel.dtf(tt)

    def generateMatrixSubTree(self):
        '''
        It takes the elements of the DT in cache and creates a matrix (N, D), where N is the number of subtrees
        :param kernel: DT
        :return: M: np.array
        '''
        return np.array([dd for k, dd in self.kernel.dtf_cache.items()])

    def createRelevanceVector(self, relevance, matrixSubTree):
        '''
        I multiply the vector of relevance with the transposed matrix of the sub-trees,
        in this way I get among all the subtrees which are the most relevant for that example.
        
        :param relevance: np.array(,D)
        :param matrixSubTree: np.array(D,N)
        :return: np.array(,N) -> the most relevant subtrees
        '''
        return np.dot(relevance, matrixSubTree)

    def relevance_matrix(self):
        '''
        From the kernel I extract all the subtrees and generate 4 dictionaries
        :param kernel: 
        :return: tree_index: {subtree: vector}
        :return: tree_index_dict: {subtree: index}
        :return: embedding_matrix: Ordered vector matrix np.array(np.array())
        :return: tree_index_dict_inverse: {index: subtree}
        '''
        tree_index = {}
        tree_index_dict = {}
        tree_index_dict_inverse = {}
        embedding_matrix = []
        count = 0
        for k, dd in self.kernel.dtf_cache.items():
            tree_index[k] = dd
            tree_index_dict[k] = count
            embedding_matrix.append(dd)
            tree_index_dict_inverse[count] = k
            count += 1
        return tree_index, tree_index_dict, np.array(embedding_matrix), tree_index_dict_inverse


    def saveActivation(self, tree, relevance_tree, tree_index_dict):
        '''
        For each tree subtree I associate the relevance value and insert it into a dictionary
        :param tree: kerMIT.Tree
        :param relevance_tree: I take the vector of relevance
        :param tree_index_dict: dict {subtree: index}
        :return: {'tree': str(tree), 'act_sub_trees':[(alberello, np.array())]}
        '''
        # per ogni sottoalbero estraggo il valore di rilevanza e lo metto in un dizionario
        return {'tree': tree, 'act_sub_trees': [[subtree, np.array([relevance_tree[tree_index_dict[subtree]]], dtype=np.float32)] for subtree in DT.subtrees(tree)[1]]}

    def activation(self, relevance, tree):
        '''

        :param relevance: np.array() dim D
        :param tree: str
        :return: ({'tree': str(tree), 'act_sub_trees':[(subtree, np.array())]},{'tree': str(tree), 'act_sub_trees':[(subtree, np.array())]})
        '''
        # calcolo i dtf dei sottoalberi di tree, cosi già ce li ho in cache
        if type(tree) is str:
            tree = Tree(string=tree)

        self.calculateDTFs(tree)

        # genero la matrice M, proprietà: M*M^T = I_nn
        m = self.generateMatrixSubTree()
        # print(m.shape, m.transpose().shape, relevance.shape)
        # mmt = np.dot(m, m.transpose())

        # a questo punto moltiplico il vettore della rilevanza per la matrice trasposta
        # prendo prima tree1
        relevance_tree = self.createRelevanceVector(relevance[:4000], m.transpose())

        # genero i 4 dizionari che mi servono per risalire ai sottoalberi
        tree_index, tree_index_dict, embedding_matrix, tree_index_dict_inverse = self.relevance_matrix()

        act_tree = self.saveActivation(tree, relevance_tree, tree_index_dict)
        self.kernel.cleanCache()

        return act_tree


    def activationRTE(self, relevance, tree1, tree2):
        '''
        It takes two trees in parenthetical form (str) as input.
        1) convert trees to kerMIT.Tree
        2) I calculate the dtf for each sub-tree of both trees
        3) I take the generated sub trees from the cache and construct a matrix (N, D) N=number of subtrees, D=dimension dtf
        4) multiply r * M^T so as to obtain a vector of dim (, N) => (, D) * (D, N) = (, N). 
        This vector tells us: how much each subtree (in the form of an index) is relevant for that input phrase
        5) from the index step to the subtree itself
        6) I return a list of pairs: (subtree, relevance)
        
        :param relevance: np.array() dim D
        :param tree1: str
        :param tree2: str
        :return: ({'tree': str(tree), 'act_sub_trees':[(subtree, np.array())]},{'tree': str(tree), 'act_sub_trees':[(subtree, np.array())]})
        '''
        # calcolo i dtf dei sottoalberi di tree1 e tree2, cosi già ce li ho in cache
        tree1 = Tree(string=tree1)
        tree2 = Tree(string=tree2)
        self.calculateDTFs(tree1)
        self.calculateDTFs(tree2)

        # genero la matrice M, proprietà: M*M^T = I_nn
        m = self.generateMatrixSubTree()
        #print(m.shape, m.transpose().shape, relevance.shape)
        #mmt = np.dot(m, m.transpose())

        # a questo punto moltiplico il vettore della rilevanza per la matrice trasposta
        # prendo prima tree1
        relevance_tree_1 = self.createRelevanceVector(relevance[:4000], m.transpose())
        # prendo poi tree2
        relevance_tree_2 = self.createRelevanceVector(relevance[4000:], m.transpose())

        # genero i 4 dizionari che mi servono per risalire ai sottoalberi
        tree_index,tree_index_dict,embedding_matrix,tree_index_dict_inverse = self.relevance_matrix()


        #print(relevance_tree_1)
        #print(relevance_tree_2)
        act_tree1 = self.saveActivation(tree1, relevance_tree_1, tree_index_dict)
        act_tree2 = self.saveActivation(tree2, relevance_tree_2, tree_index_dict)
        return act_tree1, act_tree2



    def activationQC(self, relevance, tree):
        '''

        :param relevance: np.array() dim D
        :param tree: str
        :return: ({'tree': str(tree), 'act_sub_trees':[(subtree, np.array())]},{'tree': str(tree), 'act_sub_trees':[(subtree, np.array())]})
        '''
        # calcolo i dtf dei sottoalberi di tree, cosi già ce li ho in cache
        tree = Tree(string=tree)

        self.calculateDTFs(tree)

        # genero la matrice M, proprietà: M*M^T = I_nn
        m = self.generateMatrixSubTree()
        # print(m.shape, m.transpose().shape, relevance.shape)
        # mmt = np.dot(m, m.transpose())

        # a questo punto moltiplico il vettore della rilevanza per la matrice trasposta
        # prendo prima tree1
        relevance_tree = self.createRelevanceVector(relevance[:4000], m.transpose())

        # genero i 4 dizionari che mi servono per risalire ai sottoalberi
        tree_index, tree_index_dict, embedding_matrix, tree_index_dict_inverse = self.relevance_matrix()

        act_tree = self.saveActivation(tree, relevance_tree, tree_index_dict)

        return act_tree

if __name__ == "__main__":

    kernel = DT(dimension=4000, LAMBDA=0.4, operation=op.fast_shuffled_convolution)
    interpretation = 'general' # quale modello voglio interpretare

    # apro il file con l'esempio da ispezionare ed estraggo primo tree1, tree2, relevance
    f = open("KerMIT/relevance/relevance721.txt", "r")
    readlines = f.readlines()
    tree1 = readlines[0][:-1]
    tree2 = readlines[1][:-1]
    relevance = np.array(readlines[2].split(), dtype=float)
    print("tree1", tree1, "\ntree2", tree2)
    print("relevance", relevance)

    act_lrp = ActivationSubtreeLRP(kernel)


    if interpretation == 'rte':
        act_tree1, act_tree2 = act_lrp.activationRTE(relevance, tree1, tree2)
        print("Esito:")
        print(act_tree1, act_tree2)

    elif interpretation == 'qc':
        print("qc")
        act_tree1 = act_lrp.activationQC(relevance, tree1)

        act_tree2 = act_lrp.activationQC(relevance, tree2)
        print("Esito:")
        print(act_tree1)
        print(act_tree2)
    elif interpretation == 'general':
        print(interpretation)
        act_tree1 = act_lrp.activation(relevance, tree1)

        act_tree2 = act_lrp.activation(relevance, tree2)
        print("Activation Result :")
        print(act_tree1)
        print(act_tree2)
