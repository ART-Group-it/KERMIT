__author__ = 'lorenzo'


import numpy as np
# import hashlib
# #from scipy import stats
# import time
import math
from functools import reduce
import ctypes
import sys

permutation_cache = {}

def c_mul(a, b):
    return ctypes.c_int32((int(a) * b) & 0xffffffff).value

def hash(s):
    # uses ctypes for speed
    value = ord(s[0]) << 7
    for char in s:
        value = c_mul(1000003, value) ^ ord(char)
    value = value ^ len(s)
    if value == -1:
        value = -2
    return value

def random_permutation(dimension, seed=0):
    np.random.seed(seed)
    l = np.arange(dimension)
    np.random.shuffle(l)
    return l

def random_vector(dimension, normalized=True):
    """return a random normal vector
    if normalized is true the vector is normalized, otherwise is only approximately unitary (with the variance scaled)
    """
    if normalized:
        v = np.random.normal(0, 1, dimension)

        #tentativo con numeri discreti
        #v = np.random.choice([1./np.sqrt(dimension), -1./np.sqrt(dimension)], dimension)
        norm = np.sqrt(sum(v**2))
        v = v/norm
    else:
        v = np.random.normal(0, 1./np.sqrt(dimension), dimension)

    return v

def perfect_set(dimension, number, epsilon):
    l = [random_vector(dimension)]
    for i in range(number):
        print (len(l))
        v = random_vector(dimension)
        for other_v in l:

            cos = np.dot(v, other_v)

            if not (-epsilon < cos <epsilon):
                print ("fermato a: ", len(l))
                return 0

        l.append(v)

    return l




def circular_convolution(a,b):
    f = np.fft.fft(b)
    g = np.fft.fft(a)
    z = f*g
    return np.fft.ifft(z).real

# def permutation(v):
#     h = 312   #seed
#     np.random.seed(h)
#     np.random.shuffle(v)
#     np.random.seed()
#     return v
#
# def permutation2(v):
#     h = 17912   #seed
#     np.random.seed(h)
#     np.random.shuffle(v)
#     np.random.seed()
#     return v

def fast_permutation(v, perm=None):
    if perm is None:
        perm = random_permutation(len(v))

    v = v[perm]
    return v

def fast_shuffled_convolution(v,w):
    dim = len(v)
    if dim not in permutation_cache:
        #print ("Prima volta ", dim)
        permutation_cache[dim] = [random_permutation(dim),random_permutation(dim, 123)]
        np.random.seed(11275387)

    v = fast_permutation(v,perm=permutation_cache[dim][0])
    w = fast_permutation(w,perm=permutation_cache[dim][1])

    cc = circular_convolution(v,w)
    return cc

def fast_shuffled_convolution_list(*vecs):
    return reduce(fast_shuffled_convolution(vecs))


def randomOperation(v,w):
    # makes sense only on binary vectors....
    dimension = len(v)
    norm = np.linalg.norm(v)*np.linalg.norm(w)
    patternV = [1 if x>0 else 0 for x in v]
    patternW = [1 if x>0 else 0 for x in w]

    s = np.abs(hash((bytes(patternV + patternW)))) % 4294967295
    np.random.seed(s)
    return norm*np.random.choice([-1/np.sqrt(dimension), 1/np.sqrt(dimension)], dimension)

indexCache = {}
def randomBilinearOperation(v,w):
    dimension = len(v)
    norm = np.linalg.norm(v)*np.linalg.norm(w)

    nv = []
    for i in range(dimension):
        if i in indexCache:
            j, k = indexCache[i]
            nv.append(v[j]*w[k])
        else:
            j = np.random.randint(0, dimension)
            k = np.random.randint(0, dimension)
            indexCache[i] = (j, k)
            nv.append(v[j]*w[k])
    return np.asarray(norm*nv)


    #return norm*np.random.choice([-1/np.sqrt(dimension), 1/np.sqrt(dimension)], dimension)




# def shuffled_convolution(v,w):
#     v = permutation(v)
#     w = permutation2(w)
#     cc = circular_convolution(v,w)
#     np.random.seed(s)            #inizializzare np.random.seed()
#     return cc
#     #return cc/np.sqrt(sum(cc**2))
#
# def matrix_convolution(A,B, op = shuffled_convolution):
#     shape = A.shape
#     col = shape[1]
#     C = np.zeros(shape = shape)
#     for i in range(col):
#         ca = A[:,i]
#         cb = B[:,i]
#         C[:,i] = op(ca, cb)
#     return C

if __name__ == "__main__":


    print (hash('asd'))
    #
    # DIM = 1024
    # numero = 10000
    # v = np.random.choice([-1/np.sqrt(DIM), 1/np.sqrt(DIM)], DIM)
    # w = np.random.choice([-1/np.sqrt(DIM), 1/np.sqrt(DIM)], DIM)
    #
    # c = randomBilinearOperation(v, w)
    #
    # print (np.dot(v, w))
    #
    # print (v)
    # print (w)
    # print (c)
    #
    # print ([x for x in map(np.linalg.norm, [v, w, c])])
