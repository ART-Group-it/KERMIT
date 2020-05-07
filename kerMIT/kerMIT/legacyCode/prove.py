from dtk import DT
from tree import Tree
import os
import sentence_encoder
import dataset_reader
import theano.tensor as T
from theano import function
from theano import shared
import numpy
import numpy.random as rng
import theano.gradient
import pickle

if __name__ == "__main__":


    dir = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/SVM/"
    matrix = "single-target.dm"
    file = os.path.join(dir, matrix)

    dt = DT()

    dir = "/Users/lorenzo/Documents/Universita/PHD/Lavori/DSTK/RTE"
    dev = "RTE3_dev_processed.xml.svm"
    test = "RTE3_test_processed.xml"
    dict = {"YES":1, "NO":-1}

    try:
        D = pickle.load(open("dataset_pickled", "rb"))
    except:
        print("No file found, generating one")

        Data = dataset_reader.Dataset(os.path.join(dir,dev))
        trees = [Tree(string=pairs[0]) for pairs in Data.pairs]

        D = [[],[]]
        for t in trees[1:]:
            D[0].append(dt.dt(t))
            D[1].append(sentence_encoder.encoder(t.sentence))

        D[0] = numpy.array(D[0])
        D[1] = numpy.array(D[1])

    print(D[0].shape, D[1].shape)

    #file = open("dataset_pickled", "wb")
    #pickle.dump(D, file)

    N = len(D[0]) - 1   #number of training examples

    #Theano part


    #   Declare Theano symbolic variables
    x = T.matrix("x")
    y = T.matrix("y")
    w = shared(rng.randn(1024,N), name="w")
    #b = shared(numpy.zeros(1024), name="b")
    print("Initial model:")
    print(w.get_value())

    # Construct Theano expression graph

    pred = T.dot(w,x) # + b
    cost = T.sum(((pred -y)**2),axis=1)
    cost2 = T.mean(cost)




    gw = T.grad(cost2, [w])  # Compute the gradient of the cost
    #gb = T.grad(cost, b)
    # (we shall return to this in a
    # following section of this tutorial)

    # Compile

    learning_rate = numpy.array([0.1]*1024*N)
    print(learning_rate)

    train = function(inputs=[x, y], outputs=[pred, cost], updates=[(w, w - (learning_rate* gw))])

    print("train compiled")

    predict = function(inputs=[x], outputs=pred)

    print("pred compiled")

    # Train
    training_steps = 100
    for i in range(training_steps):
        print("."),
        train(D[0], D[1])

    print("Final model:")
    print(w.get_value())
    print("target values for D:", D[1])
    print("prediction on D:", predict(D[0]))

