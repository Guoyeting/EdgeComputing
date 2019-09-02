import os
import struct
import numpy as np

def create_clients(num):
    
    num_examples = 490
    num_classes = 10

    buckets = []
    for k in range(num_classes):
        temp = []
        for j in range(num / 5):
            temp = np.hstack((temp, k * num_examples/10 + np.random.permutation(num_examples/10)))
        buckets = np.hstack((buckets, temp))
        
    shards = 2 * num
    perm = np.random.permutation(shards)
    # z will be of length 250 and each element represents a client.
    z = []
    ind_list = np.split(buckets, shards)
    for j in range(0, shards, 2):
        # each entry of z is associated to two shards. the two shards are sampled randomly by using the permutation matrix
        # perm and stacking two shards together using vstack. Each client now holds 250*2 datapoints.
        z.append(np.hstack((ind_list[int(perm[j])], ind_list[int(perm[j + 1])])))
        # shuffle the data in each element of z, so that each client doesn't have all digits stuck together.
        perm_2 = np.random.permutation(2 * len(buckets) / shards)
        z[-1] = z[-1][perm_2]

    return z

def sort(X_train, Y_train):

    train_sample = len(X_train)
    
    perm = np.random.permutation(train_sample)
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    X_train = X_train[0:490, :]
    Y_train = Y_train[0:490]

    X_train_one = X_train[:, 0]
    X_train_index = np.argsort(X_train_one)

    X_train = X_train[X_train_index, :]
    Y_train = Y_train[X_train_index]

    return X_train, Y_train

def read():
    # load train dataset
    X_train = np.array([[float(j) for j in i.rstrip().split(",")] for i in open("../train.csv").readlines()])
    Y_train = X_train[:,-1]
    X_train = X_train[:,0:-1]   
    
    # load test dataset
    X_test = np.array([[float(j) for j in i.rstrip().split(",")] for i in open("../test.csv").readlines()])
    Y_test = X_test[:,-1]
    X_test = X_test[:,0:-1]    

    return X_train, Y_train, X_test, Y_test


class Data:
    def __init__(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = read()

class Clients:
    def __init__(X_train, Y_train, client_num):
        self.X_train, self.Y_train = sort(X_train, Y_train)
        Z = create_clients(client_num)
        self.client_x = []
        self.client_y = []
        for i in range(len(Z)):
            Z[i] = Z[i].astype(np.int32)
            self.client_x.append(self.X_train[Z[i], :])
            self.client_y.append(self.Y_train[Z[i]])

