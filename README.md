# 🐸 KERMIT

KERMIT is a lightweight Python library to **encode** and **interpret** *Universal Syntactic Embeddings*

## Try It Now!
 kerMIT: [[Colab]](https://drive.google.com/file/d/1Dab_eR_c2Ko7OQUwjjgpY8vrFF3WSPaI/view?usp=sharing)

# Installation
```
git clone https://github.com/ART-Group-it/KERMIT.git 
!pip install ./KERMIT/kerMIT
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
# Citation
If you use this code, please cite:
```
@inproceedings{zanzotto-etal-2019-KERMIT,
    title = {{KERMIT: Complementing Transformer Architectures with Encoders of Explicit Syntactic Interpretations}},
    year = {2020},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    author = {Zanzotto, Fabio Massimo and Santilli, Andrea and Ranaldi, Leonardo and Onorati, Dario and Tommasino, Pierfrancesco and Fallucchi, Francesca},
    publisher = {Association for Computational Linguistics}
}
```

