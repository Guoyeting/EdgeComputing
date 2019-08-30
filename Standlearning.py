from Data_reader import Data
from Inferrence import Cancer_model
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def main():
    data = Data(os.getcwd()) 
    X_train = np.asarray(data.X_train)
    Y_train = np.asarray(data.Y_train)
    X_test = np.asarray(data.X_test)
    Y_test = np.asarray(data.Y_test)

    perm = np.random.permutation(len(X_train))
    X_train = X_train[perm]
    Y_train = Y_train[perm]

    client_number = 5
    max_sample = int(len(X_train)/client_number)*client_number
    X_train = np.split(X_train[:max_sample],client_number)
    Y_train = np.split(Y_train[:max_sample],client_number)

    #stand-alone-learning
    SAL = []
    for i in range(client_number):
        single_model = Cancer_model()
        SAL.append(single_model)
        history = single_model.fit_model(X_train[i], Y_train[i], epochs = 5000)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
    plt.title("model loss and accuracy")
    plt.ylabel("loss and accuracy")
    plt.xlabel("epoch")
    plt.legend(["train","test"],loc="center right")
    plt.savefig('result.png')

if __name__ == '__main__':
    main()
