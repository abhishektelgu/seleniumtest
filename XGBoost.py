# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 10:35:03 2018

@author: abhi_
"""

import numpy as np
import xgboost as xgb
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.datasets import dump_svmlight_file
from sklearn.externals import joblib
from sklearn.metrics import precision_score


iris=datasets.load_iris()   
X=iris.data
Y=iris.target

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

dtrain=xgb.DMatrix(X_train,label=y_train)
dtest=xgb.DMatrix(X_test,label=y_test)

# use svmlight file for xgboost -- reduce memory consumption
dump_svmlight_file(X_train, y_train, 'dtrain.svm', zero_based=True)
dump_svmlight_file(X_test, y_test, 'dtest.svm', zero_based=True)
dtrain_svm = xgb.DMatrix('dtrain.svm')
dtest_svm = xgb.DMatrix('dtest.svm')


#setting xgboost parameters
param={
        'max-depth':3, #the max depth of the tree
        'eta':0.3, # step for each iteration
        'silent':1, #quiet loggin
        'objective':'multi:softprob', #error evaluation for multiclass training
        'num_class':3} # number of classes      

num_round=20 # number of training iterations

#------- numpy array --------------
# training testing - numpy matrices
bst=xgb.train(param,dtrain,num_round)
preds=bst.predict(dtest)

# extract most confident predicitions
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Numpy array precision:", precision_score(y_test, best_preds, average='macro'))
print(best_preds)
#print("SVM file precision: ", precision_score(y_test,best_preds_svm,average="macro"))

