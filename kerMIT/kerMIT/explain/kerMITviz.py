from kerMIT.utils import visualizer as v
#DISPLAY
from IPython.display import HTML, IFrame
import json
import re
from tqdm import tqdm
from kerMIT import tree
import pathlib
import os, kerMIT

def assign_contribution_nodes(activation_SubTree):
    to_write = []
    for example in tqdm([activation_SubTree]):
        elem_dict = dict()
        activeTrees = v.activationsCalculator(example["tree"], example["act_sub_trees"])
        if(len(activeTrees.activation_level)>1):
            stringout = v.treesWithActivationsToString(activeTrees)
            for j,i in enumerate(stringout):  
                t1 = re.sub("\[","(",i)
                t1 = re.sub("\]",")",t1)
                t1 = re.sub("'","",t1)
                t1 = re.sub(":","::",t1)
                tt1 = tree.Tree(string=t1)
                for k in tt1.leaves():
                    k.root = k.root.split(":")[0] +":"+ k.root.split(":")[2]
                #    k.root = k.root.split(":")[0] + ":0" 
                elem_dict['ACT_'+str(j)] = str(tt1)
        else:
            stringout = v.treesWithActivationsToString(activeTrees)
            t1 = re.sub("\[","(",stringout[0])
            t1 = re.sub("\]",")",t1)
            t1 = re.sub("'","",t1)
            t1 = re.sub(":","::",t1)
            tt1 = tree.Tree(string=t1)
            for k in tt1.leaves():
                k.root = k.root.split(":")[0] +":"+ k.root.split(":")[2]
            elem_dict['ACT_0'] = str(tt1)
        to_write.append(elem_dict)
    return to_write


#Ricerca la path  del file index.html
def search_path():
    path_file_html = os.path.join(kerMIT.__path__[0], 'ACTree', 'tree_visualizer_pyDTE','index.html')
    script_path = str(pathlib.Path().absolute())
    path_list = script_path.split(os.sep)
    script_directory = path_list[0:len(path_list)-1]
    script_directoryp= ["../" for i in script_directory]
    rel_path = path_file_html
    path = "".join(script_directoryp) + "" + rel_path[1:]
    return path


def show_kerMITviz(heat_parse_tree):
    path_js = os.path.join(kerMIT.__path__[0], 'ACTree', 'tree_visualizer_pyDTE','heat_parse_trees','act_trees.js')
    path_html = search_path()
    #print(path_html)
    with open(path_js, 'w') as out_file:
        out_file.write('var act_trees = %s;' % json.dumps(heat_parse_tree))
    return IFrame(src=path_html, width='120%', height='500px')


