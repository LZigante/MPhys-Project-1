# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:00:22 2022

@author: Zigan
"""

from sklearn.model_selection import RandomizedSearchCV
import xgboost
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,f1_score
import pandas as pd
import numpy as np
# import seaborn as sns
from matplotlib import pyplot as plt


    
data0 = np.load('background_final100.npy',allow_pickle=True)
data1 = np.load('signal_final100.npy', allow_pickle=True)
print("data loaded!")
x_data = np.concatenate((data0, data1))
y_data = np.array([0]*len(data0)+[1]*len(data1))
# x_data, y_data = x_data/255, y_data/255
print("data combined!")

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                test_size=0.2,
                                                random_state = 27)
x_train = np.asarray(x_train).astype('float32')
x_test = np.asarray(x_test).astype('float32')


model = HistGradientBoostingClassifier(loss='log_loss', learning_rate=0.12, max_depth = 11,
                                       verbose=1, validation_fraction=0.25,random_state=27,warm_start=(True))

# params={
#     "learning_rate":[0.05,0.10,0.15,0.20,0.25,0.30],
#     "max_depth":[2,3,4,5,6,8,10,12,15]}
# clf =RandomizedSearchCV(model,param_distributions=params,n_iter=5,
#                         scoring='roc_auc',cv=5,verbose=3)

model.fit(x_train ,y_train)
# print(clf.best_estimator_)
pred = model.predict(x_test)
pred_proba=model.predict_proba(x_test)
pred_proba=pred_proba[:,1]
# t_accuracy = accuracy_score(y_train, t_prediction.round())
# v_accuracy = accuracy_score(y_test, val_pred.round())

score = model.score(x_test, y_test)
print(score)
v_accuracy = accuracy_score(y_test, pred.round())
print(v_accuracy)

f1 = f1_score(y_test, pred.round())
print(f1)
roc_auc = roc_auc_score(y_test, pred_proba)
print(roc_auc)
fpr , tpr , thresholds = roc_curve (y_test , pred_proba)
np.save('fpr_bdt',fpr)
np.save('tpr_bdt',tpr)


# best_model = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#               colsample_bynode=1, colsample_bytree=0.7,
#               enable_categorical=False, gamma=0.3, gpu_id=-1,
#               importance_type=None, interaction_constraints='',
#               learning_rate=0.3, max_delta_step=0, max_depth=2,
#               min_child_weight=7, monotone_constraints='()',
#               n_estimators=100, n_jobs=8, num_parallel_tree=1, predictor='auto',
#               random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#               subsample=1, tree_method='exact', validate_parameters=1,
#               verbosity=None)

# best_model = clf.best_estimator_

# best_model.fit(X, y)
# y_pred = best_model.predict(X)
# print('train accuracy = ',accuracy_score(y, y_pred))
# y_val_pred = best_model.predict(X_val)
# print('validation accuracy = ',accuracy_score(y_val,y_val_pred))