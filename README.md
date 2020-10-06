# 🐸 KERMIT

KERMIT is a lightweight Python library to **encode** and **interpret** *Universal Syntactic Embeddings*


<img src="img/kermit.png" alt="drawing" width="500"/>

Project Organization
------------

    |
    ├── /examples                        <- Folder containing some KERMIT examples
    │   └── /Notebooks       
    |       ├── /scripts                 <- Folder containing some scripts for our examples
    |       ├── KERMIT_encoder.ipynb     <- Jupyter Notebook for saving the KERMIT encoded trees
    |       ├── KERMIT_training.ipynb    <- Jupyter Notebook for training a system with KERMIT
    |       ├── KERMITviz.ipynb          <- Jupyter Notebook for visualizing KERMIT's heat parse trees
    |       ├── KERMITviz_Colab.ipynb    <- Jupyter Notebook for visualizing KERMIT's heat parse trees on Colab
    |       └── README.md                <- Readme file that introduces the example notebooks
    |
    ├── /kerMIT                          <- Folder containing the Python KERMIT library
    ├── /img                             <- Folder containing the images for this README file
    ├── LICENSE                          <- License file
    └── README.md                        <- This Readme file
     
--------

# Why should I use KERMIT?

- KERMIT can be used to enhance Transformers' performance on various linguistic tasks adding relevanti syntactic information from parse trees
- It is lightweight compared to a Transformer model
- KERMIT decision can be interpreted using this library and it is possible to visualize heat parse trees.

# Installation
```
git clone https://github.com/ART-Group-it/KERMIT.git 
pip install ./KERMIT/kerMIT
```
# Usage

## Demo Notebooks

- **KERMIT encoder** - Build syntactic input from a custom dataset [notebook 1](https://github.com/ART-Group-it/KERMIT/blob/master/examples/Notebooks/KERMIT_encoder.ipynb).

- **KERMIT + BERT mode**l - Train the model and save the weights [notebook 2](https://github.com/ART-Group-it/KERMIT/blob/master/examples/Notebooks/KERMIT_training.ipynb).

- **KERMITviz** - Visualize how much the syntax affects the final choice of the model [notebook 3](https://github.com/ART-Group-it/KERMIT/blob/master/examples/Notebooks/KERMITviz.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ART-Group-it/KERMIT/blob/master/examples/Notebooks/KERMITviz_Colab.ipynb)

## Quickstart with KERMIT encoder
```
from kerMIT.tree import Tree
from kerMIT.dtk import DT
from kerMIT.operation import fast_shuffled_convolution

#Insert here your parsed tree in parenthetical format 
tree = Tree(string="(A (B C))")
kermit_encoder = DT(dimension=8192, LAMBDA= 0.6, operation=fast_shuffled_convolution)

kermit_tree_encoded = kermit_encoder.dt(tree)

>> array([-0.00952759,  0.02018453, -0.02713741, ...,  0.00362533])
```


# Citation
If you use this code, please cite the paper:
```
@article{Ranaldi2020HidingYF,
  title={Hiding Your Face Is Not Enough: user identity linkage with image recognition},
  author={Leondardo Ranaldi and F. Zanzotto},
  journal={Social Network Analysis and Mining},
  year={2020},
  volume={10},
  pages={1-9}
}
```

