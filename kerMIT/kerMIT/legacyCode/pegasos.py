import numpy as np
from numpy.linalg import norm
import random
import time


#loading files
dir = "/Users/lorenzo/Downloads/svm_light_osx.8.4_i7/svm/"
train = "Dev-training.svm"
test = "Dev-testing_full_grouped.svm"

#train = open(train)
#test = open(test)

train = open(dir+train)
test = open(dir+test)





#creating dataset:
def create_dataset(train, bias=False, sampling_rate=None):
    train_d = []
    for line in train:
        l = line.split()
        y = int(l[0])
        x = np.array([float(x[2:]) for x in l[1:]])
        if bias: 
            np.insert(x,0,1)
        
        train_d.append((x,y))
    
    random.shuffle(train_d)        
        
    if sampling_rate is not None:
        try:
            pos = [item for item in train_d if item[1] == 1]
            neg = [item for item in train_d if item[1] == -1]
            print(len(pos), len(neg))
            neg = neg[:int(len(pos)*sampling_rate)]
            
            pos.extend(neg)
            train_d = pos
            random.shuffle(train_d)
            return train_d
        except Exception as e:
            print("sampling_rate too high ", e )
    
    
    
    return train_d
    



def pegasos_learner2(dataset, l=None, j=1, epoch=1, average=False):
    
    iteration = len(dataset)
    
    if l is None:
        l = 1/np.mean([norm(item[0]) for item in dataset])

    #dataset is [ ([x1, x2, ... ],y), (...),   ]
    dimension = len(dataset[0][0])             #check the format of dataset
    m = len(dataset)
    w = np.zeros(dimension)
    
    list_w = []
    for e in range(1,epoch):
        for t in range(1,iteration):
            item = dataset[t] #item = random.choice(dataset)
            x = item[0]
            y = item[1]
            eta = 1/(l*t)
            
            if y*np.dot(w, x) > 1:
                w = (1-(1/t))*w
            
            else:
                if y == 1:
                    w = (1-(1/t))*w + eta*x*j
                else:
                    w = (1-(1/t))*w - eta*x
            
            list_w.append(w)
    
    if average: 
        return np.average(list_w,axis=0)
    else: 
        return w


def classifier(dataset, w):
    dataset_dim = len(dataset)
    exact = 0
    true_pos = false_pos = 0
    true_neg = false_neg = 0
    output = []
    for item in dataset:
        x = item[0]
        y = item[1]
        if np.dot(w,x) > 0:
            pred = 1
            output.append(pred)
        else: 
            pred = -1
            output.append(pred)
        
        if pred == y:
            exact = exact + 1
            if pred == 1:
                true_pos = true_pos + 1
        
            if pred == -1:
                true_neg = true_neg + 1
        
        else:
            if pred == 1:
                false_pos = false_pos + 1
            if pred == -1:
                false_neg = false_neg + 1
    
    print(true_pos, false_pos, true_neg, false_neg)
    
    print ("accuracy: ", exact/dataset_dim)
    if true_pos + false_pos != 0:
        print ("precision positivi: ", true_pos/(true_pos + false_pos))
        print ("recall positivi: ", true_pos/(true_pos + false_neg))
        
    else:
        print ("precision: nan")
    if true_pos + false_neg != 0:
        print ("precision negativi: ", true_neg/(true_neg + false_neg))
        print ("recall negativi: ", true_neg/(true_neg + false_pos))
    else:
        print ("recall: nan")
    return output
    

train_d = create_dataset(train)
test_d = create_dataset(test)

# positivi = sum(item[1] == 1 for item in train_d)
# negativi = sum(item[1] == -1 for item in train_d)
# print("positivi:", positivi)           
# print("negativi:", negativi)
# print("ratio: ", negativi/positivi)


for jj in range(975, 985):
    j = jj/10
    w = pegasos_learner2(train_d, l=2, j=j, epoch=2)
    output = classifier(test_d, w)
    print("----")

# start_learn = time.time()
# w = pegasos_learner2(train_d, l=2, j=97.5, epoch=2)
# end_learn = time.time()
# 
# start_class = time.time()
# output = classifier(test_d, w)
# end_class = time.time()

# print("-----")
# print ("learning time: ", end_learn - start_learn)
# print ("classifying time: ", end_class - start_class)
#   
# print("----\n----")