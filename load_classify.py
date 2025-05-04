import argparse
import traceback
import time
import copy

import numpy as np
import dgl
import torch

from tgn import TGN
#from data_preprocess import TemporalWikipediaDataset, TemporalRedditDataset, TemporalDataset
from data_process import TemporalDataset
from dataloading import (FastTemporalEdgeCollator, FastTemporalSampler,
                         SimpleTemporalEdgeCollator, SimpleTemporalSampler,
                         TemporalEdgeDataLoader, TemporalSampler, TemporalEdgeCollator)

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from sklearn.model_selection import GridSearchCV

train_x = np.load('data/train_x.npy')
train_y = np.load('data/train_y.npy')
test_x = np.load('data/test_x.npy')
test_y = np.load('data/test_y.npy')
test_new_x = np.load('data/test_new_x.npy')
test_new_y = np.load('data/test_new_y.npy')
train_x_ros = np.load('data/train_x_ros.npy')
train_y_ros = np.load('data/train_y_ros.npy')
test_x_ros = np.load('data/test_x_ros.npy')
test_y_ros = np.load('data/test_y_ros.npy')
test_new_x_ros = np.load('data/test_new_x_ros.npy')
test_new_y_ros = np.load('data/test_new_y_ros.npy')
train_x_smote = np.load('data/train_x_smote.npy')
train_y_smote = np.load('data/train_y_smote.npy')
test_x_smote = np.load('data/test_x_smote.npy')
test_y_smote = np.load('data/test_y_smote.npy')
test_new_x_smote = np.load('data/test_new_x_smote.npy')
test_new_y_smote = np.load('data/test_new_y_smote.npy')
train_x_rus = np.load('data/train_x_rus.npy')
train_y_rus = np.load('data/train_y_rus.npy')
test_x_rus = np.load('data/test_x_rus.npy')
test_y_rus = np.load('data/test_y_rus.npy')
test_new_x_rus = np.load('data/test_new_x_rus.npy')
test_new_y_rus = np.load('data/test_new_y_rus.npy')
train_x_kos = np.load('data/train_x_kos.npy')
train_y_kos = np.load('data/train_y_kos.npy')
test_x_kos = np.load('data/test_x_kos.npy')
test_y_kos = np.load('data/test_y_kos.npy')
test_new_x_kos = np.load('data/test_new_x_kos.npy')
test_new_y_kos = np.load('data/test_new_y_kos.npy')

f = open('log/log_std_tgn_svc.txt', 'w')

dataset = [[train_x, train_y, test_x, test_y, test_new_x, test_new_y],
           [train_x_ros, train_y_ros, test_x_ros, test_y_ros, test_new_x_ros, test_new_y_ros],
           [train_x_smote, train_y_smote, test_x_smote, test_y_smote, test_new_x_smote, test_new_y_smote],
           [train_x_rus, train_y_rus, test_x_rus, test_y_rus, test_new_x_rus, test_new_y_rus],
           [train_x_kos, train_y_kos, test_x_kos, test_y_kos, test_new_x_kos, test_new_y_kos]]
log_content = []
for datas, labels, test_datas, test_labels, test_new_datas, test_new_labels in dataset:
    #ada = AdaBoostClassifier()
    #ada.fit(datas, labels.ravel())
    #predict_y = ada.predict(test_datas)
    #predict_new_y = ada.predict(test_new_datas)
    #predict_y_prob = ada.predict_proba(test_datas)
    #predict_y_prob = predict_y_prob[:, 1]
    #predict_new_y_prob = ada.predict_proba(test_new_datas)
    #predict_new_y_prob = predict_new_y_prob[:, 1]
    #knn = KNeighborsClassifier()
    #knn.fit(datas, labels.ravel())
    #predict_y = knn.predict(test_datas)
    #predict_new_y = knn.predict(test_new_datas)
    #predict_y_prob = knn.predict_proba(test_datas)
    #predict_y_prob = predict_y_prob[:, 1]
    #predict_new_y_prob = knn.predict_proba(test_new_datas)
    #predict_new_y_prob = predict_new_y_prob[:, 1]
    # 支持向量机模型
    svc = SVC()
    svc.fit(datas, labels.ravel())
    predict_y = svc.predict(test_datas)
    predict_new_y = svc.predict(test_new_datas)
    predict_y_prob = svc.predict_proba(test_datas)
    predict_y_prob = predict_y_prob[:, 1]
    predict_new_y_prob = svc.predict_proba(test_new_datas)
    predict_new_y_prob = predict_new_y_prob[:, 1]
    # 决策树模型
    #dtc = DecisionTreeClassifier()
    #dtc.fit(datas, labels.ravel())
    #predict_y = dtc.predict(test_datas)
    #predict_new_y = dtc.predict(test_new_datas)
    #predict_y_prob = dtc.predict_proba(test_datas)
    #predict_y_prob = predict_y_prob[:, 1]
    #predict_new_y_prob = dtc.predict_proba(test_new_datas)
    #predict_new_y_prob = predict_new_y_prob[:, 1]
    #gbdt = GradientBoostingClassifier()
    #gbdt.fit(datas, labels.ravel())
    #predict_y = gbdt.predict(test_datas)
    #predict_new_y = gbdt.predict(test_new_datas)
    #predict_y_prob = gbdt.predict_proba(test_datas)
    #predict_y_prob = predict_y_prob[:, 1]
    #predict_new_y_prob = gbdt.predict_proba(test_new_datas)
    #predict_new_y_prob = predict_new_y_prob[:, 1]
    fp_count_test = 0
    tp_count_test = 0
    tn_count_test = 0
    fn_count_test = 0
    for i in range(len(test_datas)):
        if predict_y[i] == 1 and test_labels[i] == 1:
            tp_count_test += 1
        if predict_y[i] == 1 and test_labels[i] == 0:
            fp_count_test += 1
        if predict_y[i] == 0 and test_labels[i] == 1:
            fn_count_test += 1
        if predict_y[i] == 0 and test_labels[i] == 0:
            tn_count_test += 1
    FPR_test = fp_count_test / (fp_count_test + tn_count_test)
    FNR_test = fn_count_test / (tp_count_test + fn_count_test)
    fp_count_test_new = 0
    tp_count_test_new = 0
    tn_count_test_new = 0
    fn_count_test_new = 0
    for i in range(len(test_new_datas)):
        if predict_new_y[i] == 1 and test_new_labels[i] == 1:
            tp_count_test_new += 1
        if predict_new_y[i] == 1 and test_new_labels[i] == 0:
            fp_count_test_new += 1
        if predict_new_y[i] == 0 and test_new_labels[i] == 1:
            fn_count_test_new += 1
        if predict_new_y[i] == 0 and test_new_labels[i] == 0:
            tn_count_test_new += 1
    FPR_test_new = fp_count_test_new / (fp_count_test_new + tn_count_test_new)
    FNR_test_new = fn_count_test_new / (tp_count_test_new + fn_count_test_new)
    #log_content.append("test_y:{}\n".format(test_labels))
    #log_content.append("predict_y:{}\n".format(predict_y))
    log_content.append("----------predict----------\n")
    log_content.append(
        "balanced_accuracy:{:.3f}\n f1_score:{:.3f}\n average_precision:{:.3f}\n auc:{:.3f}\n recall:{:.3f}\n FPR:{:.3f}\n FNR:{:.3f}\n".format(
            balanced_accuracy_score(test_labels, predict_y), f1_score(test_labels, predict_y), average_precision_score(test_labels, predict_y),
            roc_auc_score(test_labels, predict_y_prob), recall_score(test_labels, predict_y), FPR_test, FNR_test
        ))
    #log_content.append("test_new_y:{}\n".format(test_new_labels))
    #log_content.append("predict_new_y:{}\n".format(predict_new_y))
    log_content.append("--------predict_new--------\n")
    log_content.append(
        "balanced_accuracy:{:.3f}\n f1_score:{:.3f}\n average_precision:{:.3f}\n auc:{:.3f}\n recall:{:.3f}\n FPR:{:.3f}\n FNR:{:.3f}\n".format(
            balanced_accuracy_score(test_new_labels, predict_new_y), f1_score(test_new_labels, predict_new_y), average_precision_score(test_new_labels, predict_new_y),
            roc_auc_score(test_new_labels, predict_new_y_prob), recall_score(test_new_labels, predict_new_y), FPR_test_new, FNR_test_new
        ))
f.writelines(log_content)
print(log_content)
