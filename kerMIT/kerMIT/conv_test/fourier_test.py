__author__ = 'lorenzo'


import src.operation as op
import numpy
import matplotlib.pyplot as plt
import seaborn as sns

DIM = 1000

def fourierMatrix(n):
    return (1/numpy.sqrt(n))*numpy.fft.fft(numpy.eye(n))

def fft(v):
    n = len(v)
    return (1/numpy.sqrt(n))*numpy.fft.fft(v)


lista_n = []
lista_s = []
range_dim = range(1000,3000,500)
for DIM in range_dim:
    print (DIM)

    v = op.random_vector(DIM)
    w = op.random_vector(DIM)

    vv = fft(v)
    ww = fft(w)


    F = fourierMatrix(DIM)
    matrice = F.dot(op.fast_permutation(F.conj()))

    vett = vv * matrice.dot(ww)

    lista_n.append(numpy.linalg.norm(vv*ww))
    lista_s.append(numpy.linalg.norm(vett))
    #print (numpy.linalg.norm(vv*ww))
    #print (numpy.linalg.norm(vett))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(range_dim, lista_n)
ax.scatter(range_dim, lista_s, c="red")

plt.show()

#print (numpy.round(fourierMatrix(10).dot(numpy.conj(fourierMatrix(10))).real),1)
