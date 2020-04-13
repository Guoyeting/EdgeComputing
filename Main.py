from Data_reader import Data
from Inferrence import Cancer_model
import numpy as np
import os
from keras import backend as K
import tensorflow as tf
import random

def select_clients(train_epoch, single_epoch, client_number):
    clients = np.random.permutation(client_number)
    selected_clients_number = random.randint(1, client_number)
    selected_clients_number = min(selected_clients_number, train_epoch/single_epoch)
    return clients[:selected_clients_number]

def main():
    data = Data() 
    X_train = np.asarray(data.X_train)
    Y_train = np.asarray(data.Y_train)
    X_test = np.asarray(data.X_test)
    Y_test = np.asarray(data.Y_test)

    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    client_number = 5
    max_sample = int(len(X_train)/client_number)*client_number
    Xx_train = np.split(X_train[:max_sample],client_number)
    Yy_train = np.split(Y_train[:max_sample],client_number)
    '''
    #stand-alone-learning
    print '======= Stand alone learning ========'
    for i in range(client_number):
        single_model = Cancer_model()
        history = single_model.fit_model(Xx_train[i], Yy_train[i], epochs = 3000)
        single_model.evaluate(X_test, Y_test)
        print 'client '+ str(i)
        single_model.print_result()
        K.clear_session()
        tf.reset_default_graph()

    #centralized learning
    print '======= Centralized learning ========'
    centr_model = Cancer_model()
    history = centr_model.fit_model(X_train, Y_train, epochs = 12000)
    centr_model.evaluate(X_test, Y_test)
    centr_model.print_result()
    K.clear_session()
    tf.reset_default_graph()
    '''
    #federated learning
    print '======= Federated learning ========'
    federated_model = Cancer_model()
    SAL = [Cancer_model() for i in range(client_number)]
    train_epoch = 3000
    single_epoch = 100
    first_round = True
    while(train_epoch > 0):
        # select clients
        clients_index = select_clients(train_epoch, single_epoch, client_number)
        #clients = SAL[clients_index]
        selected_num = len(clients_index)
        train_epoch -= single_epoch*selected_num
        first_client = True
        # train clients
        for i in range(selected_num):
            if first_round == False:
                SAL[clients_index[i]].set_layer_weights('dense1', federated_model.get_layer_weights('dense1'))
                SAL[clients_index[i]].set_layer_weights('dense2', federated_model.get_layer_weights('dense2'))
                SAL[clients_index[i]].set_layer_weights('dense3', federated_model.get_layer_weights('dense3'))
            SAL[clients_index[i]].fit_model(Xx_train[clients_index[i]], Yy_train[clients_index[i]],\
                                 epochs = single_epoch)
            if first_client == True:
                layer1_weight = SAL[clients_index[i]].get_layer_weights('dense1')
                layer2_weight = SAL[clients_index[i]].get_layer_weights('dense2')
                layer3_weight = SAL[clients_index[i]].get_layer_weights('dense3')
                first_client = False
            else:
                layer1_weight_new = SAL[clients_index[i]].get_layer_weights('dense1')
                layer2_weight_new = SAL[clients_index[i]].get_layer_weights('dense2')
                layer3_weight_new = SAL[clients_index[i]].get_layer_weights('dense3')
                for j in range(2):
                    layer1_weight[j] = layer1_weight[j] + layer1_weight_new[j]
                    layer2_weight[j] = layer2_weight[j] + layer2_weight_new[j]
                    layer3_weight[j] = layer3_weight[j] + layer3_weight_new[j]
        for j in range(2):
            layer1_weight[j] = layer1_weight[j]/selected_num
            layer2_weight[j] = layer2_weight[j]/selected_num
            layer3_weight[j] = layer3_weight[j]/selected_num
           
        # federate weights
        federated_model.set_layer_weights('dense1', layer1_weight)
        federated_model.set_layer_weights('dense2', layer2_weight)
        federated_model.set_layer_weights('dense3', layer3_weight)
        first_round = False
    # evaluate
    federated_model.evaluate(X_test, Y_test)
    federated_model.print_result()
        
    K.clear_session()
    tf.reset_default_graph()        
    
if __name__ == '__main__':
    main()
