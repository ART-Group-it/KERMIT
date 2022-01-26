
from nltk.draw.util import CanvasFrame
from kerMIT.utils import tree_visualizer as tv
from kerMIT.tree import Tree
from kerMIT.dtk import DT
import random as r

def activationsVisualizer(tree,subtrees_with_activations,cf):
    ''' displays the activations of a tree by using the activations of the subtrees.
    subtrees have to be generated by the method provided in the dtk
    :param tree: the tree generating the subtrees
    :param subtrees_with_activations: a list of (subtree,activation_levels)
    where subtree is a subtree of tree and activation_levels is an array of activations for the given subtree in the network
    all the subtrees in the list should have activation_levels with the same size
    :param cf: the CanvasFrame where activations are displayed
    :return: void
    '''
    #tree_nodes_for_activation = {}
    #for node in tree.allNodes():
    #    tree_nodes_for_activation[node.id()] = node


    #for st, activations in subtrees_with_activations:
    #    for node in st.allNodes():
    #        tree_nodes_for_activation[node.id()].update_activations(activations)

    #print(tree.minmaxactivations())

    tree = activationsCalculator(tree,subtrees_with_activations)
    displ = 0
    activations_classes = len(subtrees_with_activations[0][1])
    for i in range(0,activations_classes):
        tc = tv.TreeWidget(cf.canvas(), tree, line_color="red", leaf_color='green', node_color="#f0e68c", activation=i, node_font=('helvetica', -14, 'bold'), minmax=tree.minmaxactivations() )
        cf.add_widget(tc, 10 , 10 + displ)  # (10,10) offsets
        displ = tc.bbox()[3]




def activationsCalculator(tree,subtrees_with_activations):
    tree_nodes_for_activation = {}
    for node in tree.allNodes():
        tree_nodes_for_activation[node.id()] = node

    for st, activations in subtrees_with_activations:
        for node in st.allNodes():
            tree_nodes_for_activation[node.id()].update_activations(activations, mode="max")

    #print(tree.minmaxactivations())

    return tree

def treeWithActivationsToString(tree,act_num,minmax):
    outString = "[" + tree.root + ":" + str((tree.activation_level[act_num]-minmax[0])/(minmax[1]-minmax[0]))
    if not tree.isTerminal():
        for c in tree.children:
            outString = outString + " " + treeWithActivationsToString(c,act_num,minmax)
    return outString + "]"

def treesWithActivationsToString(tree):
    minmax = tree.minmaxactivations_per_class()
    minmax_global = tree.minmaxactivations()
    sizeOfTfrees = len(tree.activation_level)
    outStrings = []
    local = False
    for i in range(0,sizeOfTfrees):
        if local:
            outStrings.append(treeWithActivationsToString(tree,i,(minmax[0][i],minmax[1][i])))
        else:
            outStrings.append(treeWithActivationsToString(tree,i,minmax_global))

    return outStrings



if __name__ == "__main__":

    cf = CanvasFrame()

    #t = Tree.fromstring('(S (NP:red this pippo) (VP (V is) (AdjP pretty)))')
    t = Tree(string='(S (NP this pippo) (VP (V is) (AdjP pretty)))')

    (_,subtrees) = DT.subtrees(t)
    r.seed(10)

    swa = [(st,[r.random(),r.random(),r.random()]) for st in subtrees]


    treeout = activationsCalculator(t,swa)

    stringout = treesWithActivationsToString(treeout)
    print(stringout)
    #activationsVisualizer(t,swa,cf)
    #cf.mainloop()

    #tree.demo()
