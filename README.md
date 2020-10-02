# 🐸 KERMIT

KERMIT is a lightweight Python library to **encode** and **interpret** *Universal Syntactic Embeddings*

![](img/kermit.png)


Project Organization
------------

    |
    ├── /examples                                   <- Folder containing some KERMIT examples
    │   └── /Notebooks       
    |       ├── /scripts                            <- Folder containing some scripts for our examples
    |       ├── KERMIT_encoder.ipynb                <- Jupyter Notebook for saving the KERMIT encoded trees
    |       ├── KERMIT_training.ipynb               <- Jupyter Notebook for training a system with KERMIT
    |       ├── KERMITviz.ipynb                     <- Jupyter Notebook for visualize KERMIT's heat parse trees
    |       └── README.md                           <- Readme file that introduces the example notebooks
    |
    ├── /kerMIT                                     <- Folder containing the Python KERMIT library
    ├── /img                                        <- Folder containing the images for this README file
    ├── LICENSE                                     <- License file
    └── README.md                                   <- This Readme file
     
--------

# Why should I use KERMIT?

1. KERMIT can be used to enhance Transformers' performance on various linguistic tasks adding relevanti syntactic information from parse trees
2. It is lightweight compared to a Transformer model
3. KERMIT decision can be interpreted using this library and it is possible to visualize heat parse trees.

# Installation
```
git clone https://github.com/ART-Group-it/KERMIT.git 
pip install ./KERMIT/kerMIT
```
# Usage


**Try it now on Google Colab!** [[Colab]](https://drive.google.com/file/d/1Dab_eR_c2Ko7OQUwjjgpY8vrFF3WSPaI/view?usp=sharing) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ART-Group-it/KERMIT/blob/master/EXPLAIN_PYTORCH-kerMIT.ipynb)

## Quickstart with KERMIT encoder
```
from kerMIT.tree import Tree
from kerMIT.dtk import DT
from kerMIT.operation import fast_shuffled_convolution

tree = Tree(string="(A (B C))")
kermit_encoder = DT(dimension=8192, LAMBDA= 0.6, operation=fast_shuffled_convolution)

kermit_tree_encoded = kermit_encoder.dt(tree)

>> array([-0.00952759,  0.02018453, -0.02713741, ...,  0.00362533,
       -0.02406953,  0.01796858])
       
```


# Citation
If you use this code, please cite the paper:
```
@inproceedings{zanzotto-etal-2019-KERMIT,
    title = {{KERMIT: Complementing Transformer Architectures with Encoders of Explicit Syntactic Interpretations}},
    year = {2020},
    booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    author = {Zanzotto, Fabio Massimo and Santilli, Andrea and Ranaldi, Leonardo and Onorati, Dario and Tommasino, Pierfrancesco and Fallucchi, Francesca},
    publisher = {Association for Computational Linguistics}
}
```

