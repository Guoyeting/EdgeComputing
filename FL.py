from keras.layers import Input, Dense
from keras.models import Model
from keras.models import clone_model
import random
from IntegratedGradients import *
import numpy as np
import time
import tensorflow as tf

def print_result(TP, TN, FP, FN, precision, recall, f_measure, accuracy):
    print 'TP, TN, FP, FN: (', TP, TN, FP, FN, '); precision: ', precision, '; recall: ', recall, '; f_measure: ', f_measure, '; accuracy: ', accuracy
    

def evaluate(X_test, Y_test, train_model):
    
    error = 0
    TP, TN, FP, FN = 0,0,0,0
    
    preds = train_model.predict(X_test)

    for i in range(len(preds)):
        if preds[i] >= 0.5:
            result = 1
        else:
            result = 0
        if result != Y_test[i]:
            error += 1
            if result == 0:
                FN += 1
            else:
                FP += 1
        if result == Y_test[i]:
            if result == 1:
                TP += 1
            else:
                TN += 1
    
    precision = float(TP)/(TP+FP)
    recall = float(TP)/(TP+FN)
    f_measure =  2*precision*recall/(precision+recall)
    accuracy = 1-float(error)/len(preds)
    
    return TP, TN, FP, FN, precision, recall, f_measure, accuracy

def train(X_train, Y_train, epochs, weight):

    train_time = time.time()
    
    inputs = Input(shape=[9])
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)

    probs = Dense(1, activation='sigmoid')(x)

    model1 = Model(inputs=inputs, outputs=probs)
    model1.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mae','acc'])

    for i in range(epochs):
        
        weight_2 = model1.get_weights()

        if weight != None:        
            for j in range(9):
                for k in range(64):
                    ns = np.random.laplace(0,20*abs(weight_2[0][j][k]-weight[0][j][k]),1)
                    #ns = np.random.laplace(0,0,1)
                    weight_2[0][j][k] += ns[0]
                
            for j in range(64):
                ns = np.random.laplace(0,20*abs(weight_2[1][j]-weight[1][j]),1)
                #ns = np.random.laplace(0,0,1)
                weight_2[1][j] += ns[0]
            
        model1.set_weights(weight_2)
        weight = model1.get_weights()
        
        model1.fit(X_train, Y_train, epochs = 1, batch_size=128, validation_split=0.15, verbose=False)            

    train_time = time.time()-train_time

    return train_time, model1

def federated_train(X_train, Y_train, single_epoch, split, epochs):

    train_time_federated = 0
    num_sample = len(Y_train)    
    # initial 
    s = 0
    start = int(float(s)/split*num_sample)
    end = int(float(s+1)/split*num_sample)
    train_time, train_model = train(X_train[start:end], Y_train[start:end], single_epoch, None)      
    weight = train_model.get_weights()
    train_time_federated += train_time        
    current_epoch = single_epoch

    
    single_model = []
    for i in range(split):
        model1 = clone_model(train_model)
        model1.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mae','acc'])
        #model1.set_weights(weight)
        single_model.append(model1)
        
    
    
    # federated
    while(current_epoch < epochs*split):
        # select
        select_num = 1
        current_epoch += (single_epoch*select_num)
        select_item = []
        weight_update = [] 
        
        while(len(select_item)!=select_num):
            item = random.randint(0,split-1)
            if item not in select_item:
                select_item.append(item)

        # distribute learning        
        for s in select_item:

            start = int(float(s)/split*num_sample)
            end = int(float(s+1)/split*num_sample)

            train_time = time.time()

            single_model[s].set_weights(weight)

            for i in range(single_epoch):

                weight_2 = single_model[s].get_weights()

                if weight != None:        
                    for j in range(9):
                        for k in range(64):
                            ns = np.random.laplace(0,20*abs(weight_2[0][j][k]-weight[0][j][k]),1)
                            #ns = np.random.laplace(0,0,1)
                            if i == 0 and weight_2[0][j][k] != weight[0][j][k]:
                                print 'error!!!!!!!' 
                            weight_2[0][j][k] += ns[0]
                           
                    for j in range(64):
                        ns = np.random.laplace(0,20*abs(weight_2[1][j]-weight[1][j]),1)
                        #ns = np.random.laplace(0,0,1)
                        if i == 0 and weight_2[1][j] != weight[1][j]:
                            print 'error!!!!!!!!!!!!'
                        weight_2[1][j] += ns[0]
                
                #if i==0 and all(weight_2) != all(weight):
                    #print 'error!!!!!!!!!!!!!!!'

                single_model[s].set_weights(weight_2)
                weight = single_model[s].get_weights()
                
                single_model[s].fit(X_train[start:end], Y_train[start:end], epochs = 1, batch_size=128, validation_split=0.15, verbose=False)            
           
            #single_model[s].set_weights(weight)
            
            #hist = single_model[s].fit(X_train[start:end], Y_train[start:end], epochs = single_epoch, batch_size=128, validation_split=0.15, verbose=False)
            
            #train_time, model1 = train(X_train[start:end], Y_train[start:end], single_epoch, weight)

            weight_update.append(single_model[s].get_weights())
            train_time_federated += (time.time()-train_time)          

        weight = weight_update[0]
        '''
        weight = weight_update[0]
        for s in range(1, select_num, 1):
            weight += weight_update[s]

        # federated
        layer_num = len(weight)
        for i in range(layer_num):
            weight[i] = weight[i]/select_num        
        '''
    train_model.set_weights(weight)

    return train_time_federated, train_model


def main():
    # load train dataset
    X_train = np.array([[float(j) for j in i.rstrip().split(",")] for i in open("train.csv").readlines()])
    Y_train = X_train[:,-1]
    X_train = X_train[:,0:-1]
    # load test dataset
    X_test = np.array([[float(j) for j in i.rstrip().split(",")] for i in open("test.csv").readlines()])
    Y_test = X_test[:,-1]
    X_test = X_test[:,0:-1]

    # repeat time
    repeat = 50
    split = 5
    # initial network
    '''
    inputs = Input(shape=[9])
    x = Dense(64, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    probs = Dense(1, activation='sigmoid')(x)
    '''
    epochs = 1000        

    # metrics
    precision_stand_alone = [0,0,0]
    recall_stand_alone = [0,0,0]
    f_measure_stand_alone = [0,0,0]
    accuracy_stand_alone = [0,0,0]

    precision_federated = 0
    recall_federated = 0
    f_measure_federated = 0
    accuracy_federated = 0

    precision_center = [0,0]
    recall_center = [0,0]
    f_measure_center = [0,0]
    accuracy_center = [0,0]

    for r in range(repeat):
        print '************ repeat: ', r, ' ***************' 
        # data preprocess
        num_sample = len(X_train)
        index = np.arange(num_sample)
        np.random.shuffle(index)
        X_train = X_train[index,:]
        Y_train = Y_train[index]

        # stand-alone learning
        precision_single = [1, 0, 0]
        recall_single = [1,0,0]
        f_measure_single = [1,0,0]
        accuracy_single = [1,0,0]

        for s in range(split):
            print 'split: ', s
            
            start = int(float(s)/split*num_sample)
            end = int(float(s+1)/split*num_sample)
            
            train_time, train_model = train(X_train[start:end], Y_train[start:end], epochs*split, None)
            TP, TN, FP, FN, precision, recall, f_measure, accuracy = evaluate(X_test, Y_test, train_model)
            print 'train time:  ', train_time
            print_result(TP, TN, FP, FN, precision, recall, f_measure, accuracy)
            
            precision_single[0] = min(precision_single[0], precision)
            precision_single[1] = max(precision_single[1], precision)
            precision_single[2] += precision/split

            recall_single[0] = min(recall_single[0], recall)
            recall_single[1] = max(recall_single[1], recall)
            recall_single[2] += recall/split

            f_measure_single[0] = min(f_measure_single[0], f_measure)
            f_measure_single[1] = max(f_measure_single[1], f_measure)
            f_measure_single[2] += f_measure/split            

            accuracy_single[0] = min(accuracy_single[0], accuracy)
            accuracy_single[1] = max(accuracy_single[1], accuracy)
            accuracy_single[2] += accuracy/split   
            
            K.clear_session()
            tf.reset_default_graph()

        for i in range(3):
            precision_stand_alone[i] += precision_single[i]
            recall_stand_alone[i] += recall_single[i]
            f_measure_stand_alone[i] += f_measure_single[i]
            accuracy_stand_alone[i] += accuracy_single[i]

        # federated learning
        train_time_federated = 0

        single_epoch = 200
        current_epoch = single_epoch

        train_time_federated, train_model = federated_train(X_train, Y_train, single_epoch, split, epochs)
        
        print 'federated result: '
        TP, TN, FP, FN, precision, recall, f_measure, accuracy = evaluate(X_test, Y_test, train_model)
        print 'train_time: ', train_time_federated/split 
        print_result(TP, TN, FP, FN, precision, recall, f_measure, accuracy)
        precision_federated += precision    
        recall_federated += recall
        f_measure_federated += f_measure
        accuracy_federated += accuracy

        K.clear_session()
        tf.reset_default_graph()        

        #central learning
        print 'central_max_iteration'
        train_time, train_model = train(X_train, Y_train, epochs*split, None)
        TP, TN, FP, FN, precision, recall, f_measure, accuracy = evaluate(X_test, Y_test, train_model)
        print 'train time:  ', train_time
        print_result(TP, TN, FP, FN, precision, recall, f_measure, accuracy)

        K.clear_session()
        tf.reset_default_graph()   

        precision_center[0] += precision
        recall_center[0] += recall
        f_measure_center[0] += f_measure
        accuracy_center[0] += accuracy
        
        print 'central_min_iteration'
        train_time, train_model = train(X_train, Y_train, epochs, None)
        TP, TN, FP, FN, precision, recall, f_measure, accuracy = evaluate(X_test, Y_test, train_model)
        print 'train time:  ', train_time
        print_result(TP, TN, FP, FN, precision, recall, f_measure, accuracy)

        K.clear_session()
        tf.reset_default_graph()   
        
        precision_center[1] += precision
        recall_center[1] += recall
        f_measure_center[1] += f_measure
        accuracy_center[1] += accuracy

    print 'comprehensive performance of single alone: '
    for i in range(3):
        precision_stand_alone[i] = precision_stand_alone[i]/repeat
        recall_stand_alone[i] = recall_stand_alone[i]/repeat
        f_measure_stand_alone[i] = f_measure_stand_alone[i]/repeat
        accuracy_stand_alone[i] = accuracy_stand_alone[i]/repeat     
    print 'precision_stand_alone: ' , precision_stand_alone
    print 'recall_stand_alone: ', recall_stand_alone
    print 'f_measure_stand_alone: ', f_measure_stand_alone
    print 'accuracy_stand_alone: ', accuracy_stand_alone

    print 'comprehensive performance of federated learning: '
    print 'precision_federated: ', precision_federated/repeat
    print 'recall_federated: ',recall_federated/repeat
    print 'f_measure_federated: ',f_measure_federated/repeat
    print 'accuracy_federated: ',accuracy_federated/repeat

    print 'comprehensive performance of center learning:'
    print 'precision_center: ', precision_center[0]/repeat, precision_center[1]/repeat
    print 'recall_center: ',recall_center[0]/repeat, recall_center[1]/repeat
    print 'f_measure_center: ',f_measure_center[0]/repeat, f_measure_center[1]/repeat
    print 'accuracy_center: ',accuracy_center[0]/repeat, accuracy_center[1]/repeat

        

if __name__ == '__main__':
    main()
    
