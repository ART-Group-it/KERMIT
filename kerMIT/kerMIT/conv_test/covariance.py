__author__ = 'lorenzo'


import kerMIT.operation
import numpy


def covariance(a, b):
    return numpy.cov(a.outer(b))


if __name__ == '__main__':
    dimension = 100
    size = 10000
    M = numpy.zeros((dimension, size))
    for i in range(size):
        a = operation.random_vector(dimension)
        b = operation.random_vector(dimension)
        c = operation.circular_convolution(a,b)
        M[:,i] = c


    C = numpy.cov(M)
    print (numpy.round(C*dimension, 2))

    print (sum(x for x in numpy.nditer(C))*dimension)       #sum of all elements
    print (numpy.trace(C*dimension))                        #trace.. should be approximately equal to dimension

    print (numpy.round(numpy.diag(C)*dimension,1))
