from Data_reader import Data
from Data_reader import Clients
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

def calculate_norm(layer_weight):
    norm = []
    for i in range(3):
        norm.append(np.sqrt(np.sum(np.square(layer_weight[i][0]))+np.sum(np.square(layer_weight[i][1]))))
    return np.asarray(norm)

def main():
    data = Data() 
    X_train = np.asarray(data.X_train)
    Y_train = np.asarray(data.Y_train)
    X_test = np.asarray(data.X_test)
    Y_test = np.asarray(data.Y_test)

    baseline1 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    baseline2 = [0, 0, 0]
    baseline3 = [0, 0, 0]
    baseline4 = [0, 0, 0]
  
    for rep in range(1):
        print '******** ', rep, ' *********'
        client_number = 10
        clients = Clients(X_train, Y_train, client_number)
        Xx_train = clients.client_x
        Yy_train = clients.client_y
        
        #stand-alone-learning
        print '======= Stand alone learning ========'
        F_measure = []
        Accuracy = []
        Train_time = []
        for i in range(client_number):
            single_model = Cancer_model()
            history, t_time = single_model.fit_model(Xx_train[i], Yy_train[i], epochs = 1000)
            single_model.evaluate(X_test, Y_test)
            print 'client '+ str(i)
            single_model.print_result()
            F_measure.append(single_model.f_measure)
            Accuracy.append(single_model.accuracy)
            Train_time.append(t_time)
            K.clear_session()
            tf.reset_default_graph()
        baseline1[0] += min(F_measure)
        baseline1[1] += max(F_measure)
        baseline1[2] += (sum(F_measure)/client_number)
        baseline1[3] += min(Accuracy)
        baseline1[4] += max(Accuracy)
        baseline1[5] += (sum(Accuracy)/client_number)
        baseline1[6] += min(Train_time)
        baseline1[7] += max(Train_time)
        baseline1[8] += (sum(Train_time)/client_number)
        
        #centralized learning
        print '======= Centralized learning ========'
        centr_model = Cancer_model()
        history,t_time = centr_model.fit_model(X_train, Y_train, epochs = 1000)
        centr_model.evaluate(X_test, Y_test)
        centr_model.print_result()
        baseline2[0] += centr_model.f_measure
        baseline2[1] += centr_model.accuracy
        baseline2[2] += t_time
        K.clear_session()
        tf.reset_default_graph()
        
        #federated learning
        print '======= Federated learning ========'
        federated_model = Cancer_model()
        federated_model_dp1 = Cancer_model()
        
        SAL = [Cancer_model() for i in range(client_number)]
        SAL_dp1 = [Cancer_model() for i in range(client_number)]

        train_epoch = 1000
        single_epoch = 5
        sigma = 0.1
        first_round = True
        federated_time = 0
        federated_time_dp1 = 0

        while(train_epoch > 0):
            # select clients
            clients_index = select_clients(train_epoch, single_epoch, client_number)
            selected_num = len(clients_index)
            train_epoch -= single_epoch*selected_num
            first_client = True
            
            clients_update = []
            clients_update_dp1 = []
            clients_norm = []
            clients_norm_dp1 = []

            # train clients
            client_train_time = []
            client_train_time_dp1 = []

            for i in range(selected_num):
                # download federated weights
                origin_weight = federated_model.get_all_layer_weights()
                origin_weight_dp1 = federated_model_dp1.get_all_layer_weights()                
                if first_round == False:
                    SAL[clients_index[i]].set_all_layer_weights(origin_weight)
                    SAL_dp1[clients_index[i]].set_all_layer_weights(origin_weight_dp1)                    
                # train model at local
                history, t_time = SAL[clients_index[i]].fit_model(Xx_train[clients_index[i]], Yy_train[clients_index[i]],\
                                     epochs = single_epoch)
                history, t_time_dp1 = SAL_dp1[clients_index[i]].fit_model(Xx_train[clients_index[i]], Yy_train[clients_index[i]],\
                                     epochs = single_epoch)
                client_train_time.append(t_time)
                client_train_time_dp1.append(t_time_dp1)
                # calculate updates
                layer_weight = SAL[clients_index[i]].get_all_layer_weights()
                layer_weight_dp1 = SAL_dp1[clients_index[i]].get_all_layer_weights()
                for k in range(3):
                    for j in range(2):
                        layer_weight[k][j] = layer_weight[k][j] - origin_weight[k][j]
                        layer_weight_dp1[k][j] = layer_weight_dp1[k][j] - origin_weight_dp1[k][j]
                clients_update.append(layer_weight)
                clients_update_dp1.append(layer_weight_dp1)
                clients_norm.append(calculate_norm(layer_weight))
                clients_norm_dp1.append(calculate_norm(layer_weight_dp1))

            S = np.median(clients_norm_dp1, axis=0)
            federated_time += max(client_train_time)
            federated_time_dp1 += max(client_train_time_dp1)
            
            # update delta
            for i in range(selected_num):
                for j in range(3):
                    if clients_norm_dp1[i][j] > S[j]:
                        for k in range(2):
                            clients_update_dp1[i][j][k] = clients_update_dp1[i][j][k]/(clients_norm_dp1[i][j]/S[j])
            
            
            # federate weights        
            layer_weight = clients_update[0]
            layer_weight_dp1 = clients_update_dp1[0]            
            for i in range(1, selected_num, 1):        
                for k in range(3):
                    for j in range(2):
                        layer_weight[k][j] += clients_update[i][k][j]
                        layer_weight_dp1[k][j] += clients_update_dp1[i][k][j]

            for k in range(3):
                for j in range(2):
                    layer_weight_dp1[k][j] = layer_weight_dp1[k][j]/selected_num + \
                                             origin_weight_dp1[k][j] +1.0/selected_num* \
                                             np.random.normal(loc=0.0, scale=float(S[k]*sigma), \
                                                              size = origin_weight_dp1[k][j].shape)
                    layer_weight[k][j] = layer_weight[k][j]/selected_num + origin_weight[k][j]
                    
            federated_model.set_all_layer_weights(layer_weight)
            federated_model_dp1.set_all_layer_weights(layer_weight_dp1)

            first_round = False
            
        # evaluate
        federated_model.evaluate(X_test, Y_test)
        federated_model.print_result()
        baseline3[0] += federated_model.f_measure
        baseline3[1] += federated_model.accuracy
        baseline3[2] += federated_time

        federated_model_dp1.evaluate(X_test, Y_test)
        federated_model_dp1.print_result()
        baseline4[0] += federated_model_dp1.f_measure
        baseline4[1] += federated_model_dp1.accuracy
        baseline4[2] += federated_time_dp1
        
        K.clear_session()
        tf.reset_default_graph()

        print 'baseline1:  ', baseline1
        print 'baseline2:  ', baseline2
        print 'baseline3:  ', baseline3
        print 'baseline4:  ', baseline4
        
        
if __name__ == '__main__':
    main()
