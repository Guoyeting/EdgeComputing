from keras.layers import Input, Dense
from keras import initializers
from keras.models import Model
from keras.models import clone_model
import random
import numpy as np
import tensorflow as tf
from Data_reader import Data, normalize_cols
import os
from keras.callbacks import Callback
import time

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.train_time = time.time()
        #self.train_time = []
        #self.begin_time = time.time()
    def on_train_end(self, logs={}):
        self.train_time = time.time()-self.train_time
        
class Cancer_model:
    
    def __init__(self):
        inputs = Input(shape=[9])
        '''
        layer1 = Dense(64, activation='relu', name = 'dense1', kernel_initializer=initializers.random_normal(stddev=0.01), bias_initializer='zeros')(inputs)
        layer2 = Dense(64, activation='relu', name = 'dense2', kernel_initializer=initializers.random_normal(stddev=0.01), bias_initializer='zeros')(layer1)
        probs  = Dense(1, activation='sigmoid', name = 'dense3', kernel_initializer=initializers.random_normal(stddev=0.01), bias_initializer='zeros')(layer2)
        '''
        layer1 = Dense(64, activation='relu', name = 'dense1')(inputs)
        layer2 = Dense(64, activation='relu', name = 'dense2')(layer1)
        probs  = Dense(1, activation='sigmoid', name = 'dense3')(layer2)
        
        self.train_model = Model(inputs=inputs, outputs=probs)
        self.train_model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['mae','acc'])
    
    def fit_model(self, X_train, Y_train, epochs = 1000, batch_size = 128, validation_split = 0.15, verbose = False):
        self.time_history = TimeHistory()
        self.history = self.train_model.fit(X_train, Y_train, epochs = epochs, batch_size = batch_size,\
                                            validation_split = validation_split, verbose = verbose,\
                                            callbacks = [self.time_history])
        return self.history, self.time_history.train_time

    def evaluate(self, X_test, Y_test):
        error = 0
        TP, TN, FP, FN = 0,0,0,0 
        preds = self.train_model.predict(X_test)
        preds = np.nan_to_num(normalize_cols(preds))
        
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
        
        self.precision = float(TP)/(TP+FP)
        self.recall = float(TP)/(TP+FN)
        self.f_measure =  2*self.precision*self.recall/(self.precision+self.recall)
        self.accuracy = 1-float(error)/len(preds)
        
        return TP, TN, FP, FN, self.precision, self.recall, self.f_measure, self.accuracy

    def print_result(self):
        print 'precision: ', self.precision, '; recall: ', self.recall, '; f_measure: ', self.f_measure, '; accuracy: ', self.accuracy

    def get_layer_weights(self, layer_name):
        return self.train_model.get_layer(layer_name).get_weights()
    def set_layer_weights(self, layer_name, weights):
        return self.train_model.get_layer(layer_name).set_weights(weights)

    def get_all_layer_weights(self):
        weights = []
        layer_name = ['dense1', 'dense2', 'dense3']
        for i in range(3):
            weights.append(self.train_model.get_layer(layer_name[i]).get_weights())
        return weights
    def set_all_layer_weights(self, weights):
        layer_name = ['dense1', 'dense2', 'dense3']
        for i in range(3):
            self.train_model.get_layer(layer_name[i]).set_weights(weights[i])       


'''

Data = Data(os.getcwd(), 5)

X_train = np.asarray(Data.X_train)
Y_train = np.asarray(Data.Y_train)
X_test = np.asarray(Data.X_test)
Y_test = np.asarray(Data.Y_test)

train_model = Cancer_model()
train_model.fit_model(X_train, Y_train, epochs = 1000, batch_size = 128, validation_split = 0.15, verbose = False)
train_model.evaluate(X_test, Y_test)
train_model.print_result() 
print train_model.get_layer_weights('dense1')

'''
