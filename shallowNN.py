# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:06:47 2023

@author: Zigan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 19:51:36 2023

@author: Zigan
"""

import numpy as np
from numpy import expand_dims
import pandas as pd

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score,f1_score,accuracy_score
import matplotlib.pyplot as plt


def get_data():
    
    data0 = np.load('background_final100.npy',allow_pickle=True)
    data1 = np.load('signal_final100.npy', allow_pickle=True)
    print("data loaded!")
    x_data = np.concatenate((data0, data1))
    y_data = np.array([0]*len(data0)+[1]*len(data1))
    # x_data, y_data = x_data/255, y_data/255
    print("data combined!")

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data,
                                                    test_size=0.4,
                                                    random_state = 27)
    x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                    test_size=0.5,
                                                    random_state = 27)
    x_train = np.asarray(x_train).astype('float32')
    x_val = np.asarray(x_val).astype('float32')


    
    return (x_train, x_val, y_train, y_val,x_test,y_test)

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


x_train, x_val, y_train, y_val,x_test,y_test = get_data()
model = Sequential()
model.add(Dense(500, input_dim = 500, activation = 'relu'))
model.add(Dense((2048), activation = 'relu'))
model.add(Dropout(0.8))
#model.add(Dense((len(X[0,:])*2), activation = 'relu'))
# model.add(Dense(4, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
print("model built!")
model.summary()

model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])
es = EarlyStopping(monitor='val_loss', patience = 3, mode = 'auto',
                   min_delta = 0, restore_best_weights=True)
history = model.fit(x_train, y_train,
                        epochs=100,
                        verbose=1, callbacks=[es],
                        validation_data = (x_val, y_val))
print("predicting values...")
# prediction = model.predict(x_train)
Y_VAL_PRED = model.predict(x_test)
# Y_VAL_PRED = pd.DataFrame(Y_VAL_PRED)
# Y_VAL_PRED = Y_VAL_PRED.drop(labels=1, axis=1)
# y_val = pd.DataFrame(y_val)
# y_val = y_val.drop(labels=1, axis=1)
acc = accuracy_score(y_test, Y_VAL_PRED.round())
print(acc)
f1 = f1_score(y_test, Y_VAL_PRED.round())
print(f1)
roc_auc = roc_auc_score(y_test, Y_VAL_PRED)
fpr , tpr , thresholds = roc_curve (y_test , Y_VAL_PRED)
np.save('fpr_sNN',fpr)
np.save('tpr_sNN',tpr)
plots(history,fpr,tpr,roc_auc)