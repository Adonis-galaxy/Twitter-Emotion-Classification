from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import sklearn
from sklearn.neighbors import KNeighborsClassifier
def validation(model,val_data,val_labels,num_val):
    label=[0,0,0,0]
    hit_label=[0,0,0,0]
    true_val=0
    predict_val = model.predict(val_data.T)
    if type(model) == MLPRegressor:
        predict_val = predict_val.argmax(1)
    for i in range(num_val):
        label[val_labels[i]] += 1
        if predict_val[i] == val_labels[i]:
            true_val += 1
            hit_label[val_labels[i]] += 1
    print("val acc:", true_val/num_val)
    print("anger prediction acc:",hit_label[0] / label[0])
    print("joy prediction acc:",hit_label[1] / label[1])
    print("optimism prediction acc:",hit_label[2] / label[2])
    print("sadness prediction acc:",hit_label[3] / label[3])
    return true_val/num_val
    