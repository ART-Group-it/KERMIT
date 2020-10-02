# How to use KERMIT

Symbolic syntactic interpretations have been central in natural language understanding.

In this work, we propose a neural network system that explicitly includes syntactic interpretations: Kernel-inspired Encoder with Recursive Mechanism for Interpretable TreesVisualizer(**KERMIT**). 

The most important result is that KERMIT allows to visualize how syntax is used in inference through KERMITviz. This system can beused in combination with transformer architectures like BERT and clarifies the use of symbolic-syntactic interpretations in specific neural networks.

Using KerMIT and kerMITviz is not difficult. KERMIT system is available at the following [link](https://github.com/ART-Group-it/kerMIT).

In this repository we provide tutorials for using KERMIT with custom datasets and the user will be able to see how the syntax is used to make inferences from the model in their context.

## Requirements
* [Transformers](https://pypi.org/project/transformers/) 2.6.0
* [KERMIT](https://github.com/ART-Group-it/kerMIT)
* [PyTorch](https://pytorch.org/) >=1.0.0 
* [stanfordcorenlp](https://stanfordnlp.github.io/CoreNLP/) 4.0.0
* [Jupyter](https://jupyter.org/install)
* [tqdm](https://pypi.org/project/tqdm/)
* [IPython](https://pypi.org/project/ipython/)

# Installation
```
git clone https://github.com/ART-Group-it/kerMIT
```
# Usage
To use KERMIT and KERMITviz with your own dataset you need specific ingredients:

* Purely syntactic input. To build the syntactic input from a custom dataset we have a guided [notebook 1](https://github.com/ART-Group-it/KERMIT/blob/master/examples/Notebooks/KERMIT_encoder.ipynb).

* BERT derived model. To train the model and save weights we offer the [notebook 2](https://github.com/ART-Group-it/KERMIT/blob/master/examples/Notebooks/KERMIT_training.ipynb).

* KERMITviz. to visualize how much the syntax affects the final choice of the model [notebook 3](https://github.com/ART-Group-it/KERMIT/blob/master/examples/Notebooks/KERMITviz.ipynb) or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ART-Group-it/KERMIT/blob/master/examples/Notebooks/KERMITviz.ipynb)


