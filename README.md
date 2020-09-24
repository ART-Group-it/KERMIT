# kerMIT

A lightweight Python library to encode and interpret Universal Syntactic Embeddings

## Try It Now!
 kerMIT: [[Colab]](https://drive.google.com/file/d/1Dab_eR_c2Ko7OQUwjjgpY8vrFF3WSPaI/view?usp=sharing)

# Installation
```
cd <folder> 
pip3 install .
```
# Usage
```
DA TERMINARE kerMIT
from kerMIT.tree import Tree
from kerMIT.dtk import DT
from kerMIT.operation import fast_shuffled_convolution

tree = Tree(string="(A (B C))")
dtCalculator = DT(dimension=8192, LAMBDA= 0.6, operation=fast_shuffled_convolution)

distributedTree = dtCalculator.dt(tree)

>> array([-0.00952759,  0.02018453, -0.02713741, ...,  0.00362533,
       -0.02406953,  0.01796858])
       
```


