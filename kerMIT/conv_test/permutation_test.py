__author__ = 'lorenzo'


import random
import numpy
import matplotlib.pyplot as plt

dimension = 1000


def shiftInPlace(l, n):
    n = n % len(l)
    head = l[:n]
    l[:n] = []
    l.extend(head)
    return l

def fixpoint(l, m):
    #returns the number of position in common between l and m
    common = 0
    lista = []
    for index, (a,b) in enumerate(zip(l,m)):
        if a == b:
            lista.append(index)
            common = common + 1
    return common




v = list(range(dimension))  #original vector
v1 = v[:]                   #2 copies to be shuffled
v2 = v[:]

random.shuffle(v1)
random.shuffle(v2)


# l = []
# n = 0
# for i in range(10000):
#     v1 = v[:]              #2 copies to be shuffled
#     v2 = v[:]
#     random.shuffle(v1)
#     random.shuffle(v2)
#     f = fixpoint(v1, v2)
#     if f:
#         n = n+1
#     l.append(f)
#
#
# print (n)
# print (numpy.mean(f))
#
# a = plt.hist(l, bins=[0,1,2,3,4,5,6,7,8,9,10])
# print (a)
# plt.show()

def lookForPerm(v):

    #v = list(range(dimension))  #original vector
    v1 = v[:]                    #2 copies to be shuffled
    v2 = v[:]

    random.shuffle(v1)
    random.shuffle(v2)

    dimension = len(v)

    n = 0
    #l = []
    for i in range(dimension):
        shiftInPlace(v1, 1)
        f = fixpoint(v1, v2)
        #print (f)
        if f > 0:
            #print (n, "F")
            return False
            #n = n+1
        #l.append(f)

    return True, v1, v2


# v = list(range(10))
# shiftInPlace(v, 1)
#
# print (v)
#
# tentativi = 1000000
# tot = 0
# for i in range(tentativi):
#     if i%100000 == 0:
#        print (i)
#     #print (".", end="")
#     v = list(range(dimension))
#     res = lookForPerm(v)
#     if res:
#         tot = tot + 1
#         print (res)
#         break



n = 0
l = []
for i in range(dimension):
    shiftInPlace(v1, 1)
    f = fixpoint(v1, v2)
    if f:
        n = n+1
    l.append(f)

print (n)

print (min(l), numpy.median(l), max(l))

a = plt.hist(l, bins=[0,1,2,3,4,5,6,7,8,9,10])
print (a)
plt.show()
