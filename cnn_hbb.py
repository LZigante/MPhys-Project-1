# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:05:59 2022

@author: Zigan
"""
'''
To do:
    - change size of network
    - add some dropout and regularization

'''
#import sys, os

import numpy as np
from numpy import expand_dims
import pandas as pd

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt


def get_data():
    
    data0 = np.load('lund_zbb550jets.npy')
    data1 = np.load('lund_zh550jets.npy')

    x_data = np.concatenate((data0, data1))
    y_data = np.array([0]*len(data0)+[1]*len(data1))
    # x_data, y_data = x_data/255, y_data/255

    x_data = np.stack(x_data)
#    y_data= np.stack(y_data)

    # print("xshape-after stack",x_data.shape)
    x_data = expand_dims(x_data, axis=3)
    y_data = keras.utils.to_categorical(y_data, 2)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data,
                                                    test_size=0.2,
                                                    random_state = 27)


    
    return (x_train, x_val, y_train, y_val)

def plots(history,fpr,tpr,roc_auc):
    #accuracy plot
    plt.figure(1)
    plt.title('Training Accuracy')
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    
    plt.figure(2)
    plt.title('ROC Curve')
    plt.plot(fpr,tpr) 
    plt.axis([0,1,0,1]) 
    plt.xlabel('False Positive Rate') 
    plt.ylabel('True Positive Rate')
    plt.text(0.99,0.01,('AUC = {:.3f}'.format(roc_auc)), ha='right',
             va='bottom',fontsize=12)
    plt.show()    

    
    #ROC
    # plt.plot(fpr, tpr, label='ROC')
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.show()
    
    return

x_train, x_val, y_train, y_val = get_data()

#Construct Network
cnnmodel = Sequential()
cnnmodel.add(Conv2D(8, (3, 3), activation='relu', input_shape=(25, 25, 1)))
cnnmodel.add(MaxPooling2D((3, 3)))
cnnmodel.add(Conv2D(16, (3, 3), activation='relu'))
cnnmodel.add(MaxPooling2D((2, 2)))
# could include dropout, regularisation, ...
cnnmodel.add(Dropout(0.25))
cnnmodel.add(Flatten())
cnnmodel.add(Dense(2, activation='softmax'))
cnnmodel.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=['accuracy'])
cnnmodel.summary()


#Train the network
es = EarlyStopping(monitor='val_loss', patience = 3, mode = 'auto',
                   min_delta = 0, restore_best_weights=True)
history = cnnmodel.fit(x_train, y_train,
                        batch_size=10000, 
                        epochs=100,
                        verbose=1, callbacks=[es],
                        validation_data = (x_val, y_val))

Y_VAL_PRED = cnnmodel.predict(x_val)
Y_VAL_PRED = pd.DataFrame(Y_VAL_PRED)
Y_VAL_PRED = Y_VAL_PRED.drop(labels=1, axis=1)
y_val = pd.DataFrame(y_val)
y_val = y_val.drop(labels=1, axis=1)

roc_auc = roc_auc_score(y_val, Y_VAL_PRED)
fpr , tpr , thresholds = roc_curve (y_val , Y_VAL_PRED)
plots(history,fpr,tpr,roc_auc)
