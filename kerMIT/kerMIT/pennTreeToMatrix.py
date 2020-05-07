__author__ = 'lorenzo'


import dtk
from tree import Tree
import numpy
import pickle
import operation
import os


def createMatrix(dtk_generator, pathToTreeBank, sections):
    #pathToTreeBank is either the path to a single file or the path to the top directory of the treeBank
    #sections is a list of string indicating the section of the ptb

    dimension = dtk_generator.dimension
    matrix = numpy.empty(shape=(0, dimension)) #create empty matrix to which append distributed vectors

    # if the path is a single file
    if os.path.isfile(pathToTreeBank):
        file = open(pathToTreeBank)
        for line in file:
            tree = Tree(string = line)
            tree.binarize()
            tree.normalize() #remove case
            distributedTree = dtk_generator.dt(tree).reshape(1, dimension) #from a dimension-vector to a (dimension x 1)-matrix
            matrix = numpy.vstack((matrix, distributedTree))

    #if the path is a directory
    else:
        for (subPath, _, files) in os.walk(path):
            currentSection = subPath[-2:]
            if currentSection in sections:
                for file in files:
                    if file.startswith('wsj'): #skip other files in the directory
                        print ("processing file: ", file)
                        file = open(os.path.join(subPath, file))
                        for line in file:
                            tree = Tree(string = line)
                            tree.binarize()
                            tree.normalize() #remove case
                            distributedTree = dtk_generator.dt(tree).reshape(1, dimension) #from a dimension-vector to a (dimension x 1)-matrix
                            matrix = numpy.vstack((matrix, distributedTree))               #append the vector to the matrix

    return matrix #each ROW is a distributed vector



if __name__ == "__main__":

    #Parameter declaration
    dimension = 8192
    LAMBDA = 0.6

    path = "/Users/lorenzo/Documents/Universita/PHD/Lavori/Datasets/PTB2"
    dtk_generator = dtk.DT(dimension=dimension, LAMBDA=LAMBDA, operation=operation.fast_shuffled_convolution)


    sections = ["0" + str(x) for x in range(10)] + [str(x) for x in range(10, 25)] #>> ["00", "01", ..., "24"]
    print (sections)

    matrix = createMatrix(dtk_generator, path, "23")

    matrixFile = open('pennTreeMatrix', 'wb')
    pickle.dump(matrix, matrixFile)

    print (matrix[1])
