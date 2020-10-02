import ast, os, pickle
from stanfordcorenlp import StanfordCoreNLP
from kerMIT.dtk import DT
from kerMIT.tree import Tree
from kerMIT.operation import fast_shuffled_convolution



#read file in pickle form
def readP(nome):
    data = []
    with open(nome, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data

#takes a tree in parenthetical form and an a filename and writes a tree in parenthetical form in a file
def writeTree(nome,tree):
    f=open(nome,'a+')
    tree = tree.replace('\xa0', ' ')
    #f.write(tree.encode("utf-8", errors="ignore"))
    f.write(tree)
    f.write('\n')
    f.close()
    
#takes a tree in parenthetical form and returns a vector of size 4000    
def createDTK(parenthetical_tree):
    
    tree = Tree(string=parenthetical_tree)
    dtCalculator = DT(dimension=4000, LAMBDA= 0.4, operation=fast_shuffled_convolution)
    distributedTree1 = dtCalculator.dt(tree=tree)
    return distributedTree1


nlp = StanfordCoreNLP(r'stanford-corenlp-full-2018-10-05')

#takes sentences and parse freetext in parenthetical tree
def parse(text):
    
    text = (text.encode('ascii', 'ignore')).decode("utf-8")

    try:
        try:
            parsed=""
            props={'annotators': 'parse','outputFormat':'json'}
            output = nlp.annotate(text, properties=props)
        except Exception:
            return "(S)"
        
        outputD = ast.literal_eval(output)
        senteces = outputD['sentences']     
        #check if there are more than one sentence
        if len(senteces) <= 1:
            root = senteces[0]['parse'].strip('\n')
            root = root.split(' ',1)[1]
            root = root[1:len(root)-1]
        else:
            s1 = senteces[0]['parse'].strip('\n')
            s1 = s1.split(' ', 1)[1]
            s1 = s1[1:len(s1)-1]
            root = "(S" + s1
        #split sentences
            for sentence in senteces[1:]: 
                s2 = sentence['parse'].strip('\n')
                s2 = s2.split(' ', 1)[1]
                s2 = s2[1:len(s2)-1]
                root = root + s2
            root = root + ")"
        return root.replace("\n", "")
    except Exception:
        return "(S)"
