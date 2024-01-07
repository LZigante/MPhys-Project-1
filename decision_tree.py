# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:26:21 2022

@author: Zigan
"""

import tensorflow as tf
#import tensorflow_decision_forests as tfdf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import math
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,f1_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import tree

def plots(t,v,fpr,tpr,roc_auc):
    #accuracy plot
    plt.figure(1)
    plt.title('Training Accuracy')
    plt.plot(t, label='accuracy')
    plt.plot(v, label = 'validation accuracy')
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
# DF_B = pd.read_csv("vh_chw_zero_background.dat",sep='\s+',engine='python')
# DF_S = pd.read_csv("vh_chw_zp1_signal.dat",sep='\s+',engine='python')
# DF_TOT = DF_S.append(DF_B)
# DF_MIX = DF_TOT.sample(frac=1).reset_index(drop=True)
data0 = np.load('background_final100.npy',allow_pickle=True)
data1 = np.load('signal_final100.npy', allow_pickle=True)
print("data loaded!")
x_data = np.concatenate((data0, data1))
y_data = np.array([0]*len(data0)+[1]*len(data1))
# x_data, y_data = x_data/255, y_data/255
print("data combined!")
#x_data = SelectKBest(chi2, k=350).fit_transform(x_data, y_data)
x_train, x_val, y_train, y_val = train_test_split(x_data, y_data,
                                                test_size=0.4,
                                                random_state = 27)
x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,
                                                 test_size=0.5,
                                                 random_state = 27)
x_train = np.asarray(x_train).astype('float32')
x_val = np.asarray(x_val).astype('float32')
x_test = np.asarray(x_test).astype('float32')

# train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(DF_MIX)
# DF_MIX = np.array(DF_MIX)
# X = DF_MIX[:,:-1]
# Y = DF_MIX[:,-1]
d=7
sc = np.array([])
    
model_1 = DecisionTreeClassifier(criterion='entropy', max_depth = d)

model_1.fit(x_train,y_train)
# t_prediction = model_1.predict(x_train)
# v_prediction = model_1.predict(x_val)
val_pred = model_1.predict(x_test)
y_pred=model_1.predict_proba(x_test)[:,1]
# t_accuracy = accuracy_score(y_train, t_prediction.round())
v_accuracy = accuracy_score(y_test, val_pred.round())
f1 = f1_score(y_test, val_pred.round())
score = model_1.score(x_test, y_test)
print("done")

# plt.figure(1)
# plt.title('Accuracy score vs Max depth')
# plt.plot(D,sc)
# plt.xlabel('Max depth')
# plt.ylabel('Accuracy score')
# #plt.legend(loc='lower right')
# plt.show()
#print(score)
# val_pred = model_1.predict_proba(x_val)
# print(t_accuracy)
# print(v_accuracy)

roc_auc = roc_auc_score(y_test, y_pred)
fpr , tpr , thresholds = roc_curve (y_test , y_pred)
np.save('fpr_dttest',fpr)
np.save('tpr_dttest',tpr)
#fpr , tpr , thresholds = roc_curve (y_val , val_pred)
#plots(t_accuracy, v_accuracy,fpr,tpr,roc_auc)
#prediction = model_1.predict(x)
# y = prediction.round()
# accuracy = accuracy_score(Y, prediction.round())
# print('Accuracy =', accuracy)

# model_2 = tfdf.keras.GradientBoostedTreesModel(num_trees = 1000,
#                                                        max_depth = 8)
# model_2.fit(train_ds)


#clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
#clf_en.fit(X_train, y_train)
#y_pred_en = clf_en.predict(X_test)