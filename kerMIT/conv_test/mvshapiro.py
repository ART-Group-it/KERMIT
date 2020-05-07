__author__ = 'lorenzo'

import numpy
from numpy.linalg import inv
from scipy.linalg import sqrtm
from scipy.stats import shapiro, norm



def multiVariateShapiro(data):
    # data is a matrix
    # rows are observations, columns are variable

    # compute sample mean and covariance matrix
    sampleMean = numpy.mean(data, axis=0)
    sampleCovariance = numpy.cov(data, rowvar=0, bias=0)
    centeredData = data - sampleMean

    S = sqrtm(inv(sampleCovariance))                     # vedere come calcolare più efficientemente
    transformedData = numpy.dot(S, centeredData.T)

    univariateShapiro = []
    for i in transformedData:
        W, _ = shapiro(i)
        univariateShapiro.append(W)
    t = numpy.mean(univariateShapiro)
    return t



def computeValue(data):
    size, dimension = data.shape

    W = multiVariateShapiro(data)
    y = numpy.log(size)
    W1 = numpy.log(1 - W)
    m = -1.5861 - 0.31082*y - 0.083751*y**2 + 0.0038915*y**3
    s = numpy.exp(-0.4803 - 0.082676 * y + 0.0030302 * y**2)
    s2 = s**2
    sigma2 = numpy.log((dimension - 1 + numpy.exp(s2))/dimension)
    mu1 = m + s2/2 - sigma2/2
    pValue = norm.sf(W1, loc=mu1, scale=numpy.sqrt(sigma2))
    return W, pValue



if __name__ == "__main__":


    dataset = numpy.genfromtxt("dataset10_1.csv", delimiter="\t")

    l = []
    for i in range(20):
        uniformVectors = numpy.random.normal(0,1,(2000, 1024))
        l.append(computeValue(uniformVectors))

    print (numpy.mean(l))

    #print (computeValue(dataset))
