from collections import defaultdict, Counter
from functools import reduce
import pickle
import random
import csv
import numpy
import math
import functools
import itertools
import matplotlib
# matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns

import scipy.stats
import scipy.linalg
from scipy.stats import shapiro, norm

from kerMIT.mvshapiro import multiVariateShapiro, computeValue
from kerMIT import operation as op




def replacement(N, n_repl):
    G = list(range(N))
    G_ = list(sorted(numpy.random.choice(G, n_repl)))
    counts = [G_.count(i) for i in G]
    return G_, counts


def conv_with_repetition(num_vettori, num_ripetuti, dimension, operation, distribution="norm"):
    l = random_list_vec(num_vettori, num_ripetuti, dimension)
    nl = []
    v = l[0]
    nl.append(v)
    for i in l[1:]:
        #print("..")
        v = operation(v,i)
        nl.append(v)

    return nl


def random_vec(dimension, distribution="norm", normalized=True):

    if distribution=="norm":
        numpy.random.seed()
        v = numpy.random.normal(0, 1/numpy.sqrt(dimension), dimension)

    if distribution=="binary":
        v = numpy.random.choice([+1,-1],size=dimension)

    if normalized==True:
        v = v/numpy.linalg.norm(v)

    return v


def random_list_vec(num_vettori, num_ripetuti, dimension, distribution="norm", normalized=True):
    l = []
    for i in range(num_vettori - num_ripetuti):
        l.append(random_vec(dimension,distribution,normalized))
    v = random_vec(dimension, distribution, normalized=normalized)
    for j in range(num_ripetuti):
        l.append(v)

    #numpy.random.shuffle(l)  #optional...vedere se fa differenza
    return l


def sample(num_sample, num_vettori, num_ripetuti, dimension, operation, distribution="norm",last=True):
    if last:
        #returns only the final convolution
        L = [conv_with_repetition(num_vettori,num_ripetuti,dimension,operation, distribution)[-1] for i in range(num_sample)]
    else:
        #returns all the convolutions
        L = [conv_with_repetition(num_vettori,num_ripetuti,dimension,operation, distribution) for i in range(num_sample)]
    return L


def sample_variance(sample_size, dimension, num_vettori, operation, plot=False):
    l = []
    for i in range(sample_size):

        v = numpy.random.normal(0, 1/numpy.sqrt(dimension), dimension)
        v = v/numpy.linalg.norm(v)

        for i in range(num_vettori):

            numpy.random.seed()         #in questo modo dovrei aumentare la randomicità....

            w = numpy.random.normal(0, 1/numpy.sqrt(dimension), dimension)
            w = w/numpy.linalg.norm(w)
            v = operation(v,w)          #seed che si resetta dovrebbe essere apposto

        l.append(v)

    return numpy.var(l,axis=0,ddof=1)


def test_distribution(num_vettori, dimensioni, operation, plot=False, distribution="norm"):

    for dimension in dimensioni:

        #v = numpy.random.normal(0, 1/numpy.sqrt(dimension), dimension)
        #v = v/numpy.linalg.norm(v)

        L = sample(1, num_vettori, 0, dimension, operation)

        #print (L)
        v = L[0]

        x = numpy.linspace(-4*numpy.std(v), 4*numpy.std(v))
        x = numpy.linspace(-4/numpy.sqrt(dimension), 4/numpy.sqrt(dimension))

        print (1/numpy.sqrt(dimension))
        print (numpy.std(v))
        print (numpy.linalg.norm(v))

        if plot:
            plt.plot(x, scipy.stats.norm.pdf(x,scale=1/numpy.sqrt(dimension)),'r-', alpha=0.6, label='norm pdf')
            plt.plot(x, scipy.stats.norm.pdf(x,scale=numpy.std(v)),'g-', alpha=0.6, label='norm pdf')
            plt.hist(v, bins=20, normed=True)#,color="#3F5D7D")

            #plt.legend(["mean:\t " + str(numpy.mean(v))])

            plt.annotate("mean: " + str(numpy.mean(v)), xy=(0.05, 0.95), xycoords='axes fraction')
            plt.annotate("variance: " + str(numpy.var(v)), xy=(0.05, 0.85), xycoords='axes fraction')
            plt.show()

            op_name = operation.__name__
            file_name = op_name + "num_conv_" + str(num_vettori) + "dim" + str(dimension)
            plt.savefig("conv_test/test_distribution/MORESEED" + file_name + ".png")

            plt.clf()
        #return numpy.std(v,ddof=1)

    return v


def normality_test(num_vettori, dimensioni, operation, distribution="norm"):
    for dimension in dimensioni:
        v = random_vec(dimension,distribution,True)
        l_k = []
        for i in range(num_vettori):
            numpy.random.seed()         #in questo modo dovrei aumentare la randomicità....
            w = random_vec(dimension,distribution,True)
            v = operation(v,w)          #seed che si resetta dovrebbe essere apposto
            #l_k.append(scipy.stats.kstest(v, "norm", args=(0, 1/numpy.sqrt(dimensioni)))[1])
            l_k.append(scipy.stats.shapiro(v)[1])

        plt.plot(l_k)
        plt.show()
        op_name = operation.__name__
        file_name = "num_conv_" + str(num_vettori) + "dim_" + str(dimension) + op_name

        plt.plot([0.05]*num_vettori)
        plt.plot([0.01]*num_vettori, "r")

        plt.savefig("conv_test/test_normality/shapiro_" + file_name + ".png")
        plt.clf()

        scipy.stats.probplot(v,plot=plt)
        #plt.show()
        plt.savefig("conv_test/test_normality/prob_plot_" + file_name + ".png")
        plt.clf()


def normality_test_component_wise(tentativi, num_vettori, dimensioni, operation, distribution="norm"):
    for dimension in dimensioni:
        samples = []
        for t in range(tentativi):
            v = random_vec(dimension,distribution,True)

            for i in range(num_vettori):
                numpy.random.seed()         #in questo modo dovrei aumentare la randomicità....
                w = random_vec(dimension,distribution,True)
                v = operation(v,w)          #seed che si resetta dovrebbe essere apposto
            samples.append(v)

        c = numpy.array(samples).transpose()

        shap = []
        for i in c:
            s = scipy.stats.shapiro(i)[1]
            shap.append(s)



        plt.plot(shap)
        plt.show()
        op_name = operation.__name__
        file_name = "num_conv_" + str(num_vettori) + "dim_" + str(dimension) + op_name
        plt.plot([0.05]*dimension)
        plt.plot([0.01]*dimension, "r")

        plt.savefig("conv_test/test_normality/component_wise" + str(tentativi) + file_name + ".png")
        plt.clf()


def test_norme(tentativi, num_vettori, dimensioni, operation,num_ripetuti=0,plot_type="var"):
    for dimension in dimensioni:
        L = []
        for t in range(tentativi):

            l = conv_with_repetition(num_vettori,num_ripetuti,dimensioni,operation)
            lista_norme = [numpy.linalg.norm(v) for v in l]

            L.append(lista_norme)


        #il plot lo metto dentro o fuori dalla funzione ??
        #return L
        if plot_type == "var":
            a, mi, ma = numpy.mean(L,axis=0), numpy.min(L, axis=0), numpy.max(L, axis=0)
            std = numpy.std(L, axis=0)

            #PLOTTING
            #ax = plt.subplot(111)
            #ax.spines["top"].set_visible(False)
            #ax.spines["right"].set_visible(False)

            #ax.get_xaxis().tick_bottom()
            #ax.get_yaxis().tick_left()

            plt.xlabel("number of convolutions")
            plt.ylabel("average norm")

            #plt.fill_between(range(num_vettori), a+std, a-std,color="#3F5D7D")
            plt.plot(range(num_vettori), a)#, color="white")

            plt.show()

            #codice per salvare immagini
            op_name = operation.__name__
            file_name =  op_name + str(dimension) + "sample" + str(tentativi) + "conv" + str(num_vettori) + "rip" + str(num_ripetuti)
            plt.savefig("conv_test/test_norme/MORESEED" + file_name + ".png")
            plt.clf()


def scalar_product(G):
    """
    given a set normalVectors of vectors, computes dot products between every pair of different vectors
    """
    prodotti = []
    for x, y in itertools.combinations(G, 2):
        prodotti.append(x.dot(y))
    #sns.distplot(prodotti, kde=False, fit=scipy.stats.norm)
    return prodotti


def compute_norme(G):
    lista_norme = [numpy.linalg.norm(x) for x in G]
    #sns.distplot(lista_norme, kde=False, fit=scipy.stats.norm)
    return lista_norme


def scalar_product_and_norm(G):
    """
    given a set normalVectors of vectors, computes dot products between every pair of different vectors
    TODO: add statistics to the plot such as mean, std and whatever.
    """
    prodotti = []
    n = len(G) + 1
    print ("iterazioni da fare: ", n**2)

    for i, (x, y) in enumerate(itertools.product(G, repeat=2)):
        if i % 100000 == 0:
            print (i, end=" ")
        if i % n == 0:
            prodotti.append(x.dot(y) - 1)
        else:
            prodotti.append(x.dot(y))

    #plt.hist(prodotti, normed=True, bins=100)


    # sns.distplot(prodotti, kde=False, fit=scipy.stats.norm)
    print ("\n")
    return prodotti


def covarianceDistance(G):
    # G is a list of vectors
    dimension = len(G[0]) # dimension of first vector
    sqrtDimension = numpy.sqrt(dimension)
    covarianceMatrix = numpy.cov(G, rowvar=0)
    I = numpy.eye(dimension)/dimension
    distance = numpy.linalg.norm(covarianceMatrix - I)
    trace = numpy.trace(covarianceMatrix - I)
    return distance, trace

def wassersteinDistance(G):
    dimension = len(G[0]) # dimension of first vector
    dimension_sqrt = numpy.sqrt(dimension)
    A = numpy.cov(G, rowvar=0)
    B = numpy.eye(dimension)/dimension
    A_sqrt = scipy.linalg.sqrtm(A)
    # ABA = scipy.linalg.sqrtm(A_sqrt.dot(B).dot(A_sqrt))
    dd = numpy.trace(A + B - 2*(A_sqrt)/dimension_sqrt)
    d = numpy.sqrt(dd)
    return d.real

def conv_with_replacement(n_vettori, dimension, n_conv, operation = op.fast_shuffled_convolution):
    G = list(zip(random_list_vec(n_vettori, 0, dimension), [0]*n_vettori))
    # orders = defaultdict(int)
    if n_conv == 0:
        return G
    pairs_taken = []
    N = n_vettori
    # orders = dict(zip(range(numBaseVectors), [0]*numBaseVectors))

    for i in range(n_conv):
        i1 = numpy.random.choice(N)
        i2 = numpy.random.choice(N)
        #print (i1, i2)
        #order = orders[i1] + orders[i2] + 1
        if (i1, i2) not in pairs_taken:
            a_, order_a = G[i1]
            b_, order_b = G[i2]
            c = operation(a_, b_)
            G.append((c, order_a + order_b + 1))
            N = N + 1
            pairs_taken.append((i1, i2))
            #orders[numBaseVectors] = order
    #print (numBaseVectors)
    #print (pairs_taken)

    return G


def prob(G, center = 0, eps = 0.1):
    m, M = center - eps, center + eps
    return len([x for x in G if m <= x <= M])/len(G)


def create_set(numero_vettori, ordine, dimension, operation = op.fast_shuffled_convolution):
    # credo di poterlo modificare usando random.sample .....
    pairs_taken = []
    G = {} #key=ordine, value = lista di vettori di quell'ordine
    if ordine == 0:
        G[0] = random_list_vec(numero_vettori, 0, dimension, normalized=False)
        #ritrasformare in un unica lista?
        return G
    else:
        G[0] = random_list_vec(numero_vettori, 0, dimension, normalized=False)

        for k in range(1, ordine):
            l = []
            for d in range(numero_vettori):
                i = random.choice(range(k)) #choose random index for the set G_i
                j = k - i - 1

                i_a = numpy.random.choice(len(G[i]))
                i_b = numpy.random.choice(len(G[j]))
                if (i, j, i_a, i_b) not in pairs_taken:
                    pairs_taken.append((i, j, i_a, i_b))
                    v_a = G[i][i_a]
                    v_b = G[j][i_b]
                    #v_a = random.choice(normalVectors[i])
                    #v_b = random.choice(normalVectors[j])
                    c = operation(v_a, v_b)
                    l.append(c)

            G[k] = l

    return G


def vectorsUpToN(G, sample=None):
    # returns a dict of key=n, value = list of vector up to degree n
    vectors = {0: G[0]}
    N = len(G)
    for i in range(1, N):
        vectors[i] = vectors[i-1] + G[i]
        if sample and sample < len(vectors[i]):
            vectors[i] = random.sample(vectors[i], sample)
    return vectors


def loadOrCreateDataset(numBaseVectors, dimension, numConvolutions, sample=None):
    #try loading an existing dataset, otherwise creates it
    fileName = "vectors_{0}_{1}_{2}".format(numBaseVectors, dimension, numConvolutions)

    try:
        setOfConvolutions_dict = pickle.load(open(fileName, "rb"))
        print ("dataset loaded: ", fileName)
    except FileNotFoundError:
        print ("creating dataset")
        # sets generation
        setOfConvolutions_dict = create_set(numBaseVectors, numConvolutions, dimension) #dict of list of vectors, keyed by degree of convolutions
        pickle.dump(setOfConvolutions_dict, open(fileName, "wb"))

    return vectorsUpToN(setOfConvolutions_dict, sample)


def dictToList(G, n=None):
    if n is None:
        n = len(G)
    l = []
    for i in range(n):
        l.extend(G[i])
    return l


def plotDataset(vectorsUpToN, mode, epsilons):
    # get parameters from convolutionsDict
    numBaseVectors = len(vectorsUpToN[0])
    dimension = len(vectorsUpToN[1][0])
    numConvolutions = len(vectorsUpToN)

    scalarProductsList = None
    normsList = None

    if "sp" in mode: #scalar product
        plt.clf()
        fileName = "scalarProductsList{0}_{1}_{2}".format(numBaseVectors, dimension, numConvolutions)
        try:
            scalarProductsList = pickle.load(open(fileName,"rb"))
            print ("scalar products loaded:", fileName)
        except FileNotFoundError:
            print ("computing and storing scalar products")
            scalarProductsList = []
            for n in range(0, numConvolutions):
                vectors = vectorsUpToN[n]

                # decidere quanti sample
                # if len(vectors) > 1000:
                #     vectors = random.sample(vectors, 1000)

                scalarProductsList.append(scalar_product(vectors))
            pickle.dump(scalarProductsList, open(fileName, "wb"))

        # scalar products decreasing performance
        decadimento_sp = []

        for scalarProducts in scalarProductsList:
            probabilities = [prob(scalarProducts, center = 0, eps = x) for x in epsilons]
            decadimento_sp.append(probabilities)



        # plotting
        for dn in zip(*decadimento_sp):
            plt.plot(range(0, numConvolutions), dn)
        #TODO: add labels to the plot

        plt.xlabel("number of convolutions", fontsize=15)
        plt.ylabel("probability", fontsize=15)

        plt.show()
        print ("sp: ", decadimento_sp)
    if "norm" in mode:
        plt.clf()
        normsList = []
        for n in range(0, numConvolutions):
            normsList.append(compute_norme(vectorsUpToN[n]))

        decadimento_norme = []
        for norms in normsList:
            probabilities = [prob(norms, center = 1, eps = x) for x in epsilons]
            decadimento_norme.append(probabilities)

        for dn in zip(*decadimento_norme):
            plt.plot(range(0, numConvolutions), dn)
        plt.xlabel("number of convolutions", fontsize=15)
        plt.ylabel("probability", fontsize=15)
        plt.show()
        print ("norme: ", decadimento_norme)


    return scalarProductsList, normsList, decadimento_norme, decadimento_sp


def shapiroTest(vectorsUpToN):
    shapiroes = []
    for i in range(0, numConvolutions):
        #_, pvalue = computeValue(numpy.asarray(vectorsUpToN[i]))
        #print (_, pvalue)
        #shapiroes.append(pvalue)
        print ("cov: ", covarianceDistance(vectorsUpToN[i]))
    plt.plot(range(1, numConvolutions), shapiroes)
    plt.show()



if __name__ == '__main__':

    sns.set_context("paper")


    # parameter declaration
    numBaseVectors = 100 #initial number
    dimension = 32000     #dimension
    numConvolutions = 50 #max_order
    sample = 200

    epsilons = [0.1, 0.05, 0.04, 0.03, 0.02, 0.01]

    #try loading an existing dataset, otherwise creates it
    vectorsUpToN = loadOrCreateDataset(numBaseVectors, dimension, numConvolutions, sample)


    # #shapiro test
    # shapiroes = []
    # for i in range(1, numConvolutions):
    #     _, pvalue = computeValue(numpy.asarray(vectorsUpToN[i]))
    #     #print (_, pvalue)
    #     if pvalue > 0.05:
    #         shapiroes.append(1)
    #     else:
    #         shapiroes.append(0)
    #
    #     print ("cov: ", covarianceDistance(vectorsUpToN[i]))
    # plt.plot(range(1, numConvolutions), shapiroes)
    # plt.show()



    # computing and plotting scalar products and norms up to N
    scalarProductsList, normsList, decadimento_norme, decadimento_sp = plotDataset(vectorsUpToN, "norm sp", epsilons)

    print ("nconv\teps01\teps005\teps004\teps003\teps002\teps001")
    for i, l in list(enumerate(decadimento_sp))[::5]:
        print (i, "\t", "\t".join(str(x) for x in l))


    print ("nconv\teps01\teps005\teps004\teps003\teps002\teps001")
    for i, l in list(enumerate(decadimento_norme))[::5]:
        print (i, "\t", "\t".join(str(x) for x in l))


    # univariate shapiro test on scalar product
    # shapiroes = []
    # for sp in scalarProductsList:
    #     W, p = shapiro(sp)
    #     print (W, p)
    #     shapiroes.append(p)
    #     #print ("shapiro:", multiVariateShapiro(vectorsUpToN[n]))
    # print (len(shapiroes))
    # plt.plot(range(1, numConvolutions - 1), shapiroes)
    # plt.show()





    #comparison between scalar products of normal and convolved vectors
    # convolvedVevtors = vectorsUpToN[numConvolutions - 1]
    # #fast enough not to require pickling it
    # sampleSize = 1000
    # normalVectors = random_list_vec(sampleSize, 0, dimension, normalized=False)
    # normalScalarProducts = scalar_product(normalVectors)
    #
    # # plotting on the same pic
    # ax1 = plt.subplot(2, 1, 1)
    # sns.distplot(scalarProductsList[-1])
    # ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    # sns.distplot(normalScalarProducts)
    # plt.show()
