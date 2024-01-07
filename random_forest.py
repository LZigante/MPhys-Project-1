# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 13:29:23 2022

@author: Zigan
"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,f1_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
def read_in_data():
    data0 = np.load('background_final100.npy',allow_pickle=True)
    data1 = np.load('signal_final100.npy', allow_pickle=True)
    print("data loaded!")
    x_data = np.concatenate((data0, data1))
    y_data = np.array([0]*len(data0)+[1]*len(data1))
    #x_data = SelectKBest(chi2, k=350).fit_transform(x_data, y_data)
    # x_data, y_data = x_data/255, y_data/255
    print("data combined!")
    
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data,
                                                    test_size=0.2,
                                                    random_state = 27)
    x_train = np.asarray(x_train).astype('float32')
    x_val = np.asarray(x_val).astype('float32')

    # X = df_mix[:,:-1]
    # Y = df_mix[:,-1]
    # n = df_mix.shape[0]
    # n = int(n)
    # n_train = int(0.8 * n)
    # X_train = X[n_train:,:]
    # X_val = X[:(1-n_train),:]
    # y_train = Y[n_train:]
    # y_val = Y[:(1-n_train)]
    return(x_train, x_val, y_train, y_val)
N = [200]
S = [10]
sc = np.array([])
rc = np.array([])

X_train, X_val, y_train, y_val = read_in_data()
rfc = RandomForestClassifier(n_estimators = 1, criterion = "log_loss",
                             verbose = 2,max_depth=7
                             ,max_samples=0.35)

rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_val)
y_pred_proba = rfc.predict_proba(X_val)[:,1]
v_accuracy = accuracy_score(y_val, y_pred.round())
f1 = f1_score(y_val, y_pred.round())
score = rfc.score(X_val, y_val)
roc_auc = roc_auc_score(y_val, y_pred_proba)
fpr , tpr , thresholds = roc_curve (y_val , y_pred_proba)
np.save('fpr_rftest',fpr)
np.save('tpr_rftest',tpr)
print('accuracy = ', v_accuracy)
print('AUC = ', roc_auc)


