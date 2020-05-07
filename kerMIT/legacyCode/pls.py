__author__ = 'lorenzo'

from sklearn.pls import PLSCanonical, PLSRegression, CCA
import pickle
import numpy as np
from dataset_creator import DatasetCreator


params={"LAMBDA":0.4, "dimension":4096}

c = DatasetCreator(dtk_params = params, encoder_params=[4096, 3])

D = c.get_d()

n = len(D[0])

print(D[1])


train_X, train_Y = D[0][:n/2], D[1][:n/2]
test_X, test_Y = D[0][n/2:], D[1][n/2:]



pls2 = PLSRegression()
pls2.fit(train_X, train_Y)

#print(pls2.coefs)

pred = pls2.predict(test_X)

mean_err =np.mean((pred - test_Y)**2)

print(mean_err)


mean_cos = 0
mean_cos_original = 0

for i,j in zip(pred, test_Y):
    mean_cos = mean_cos + np.dot(i,j)/np.sqrt(np.dot(i,i)*np.dot(j,j))


print(mean_cos/n)
