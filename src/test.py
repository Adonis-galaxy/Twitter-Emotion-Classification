from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
import sklearn
from sklearn.neighbors import KNeighborsClassifier
def testing(model,test_data,test_labels,num_test):
    label=[0,0,0,0]
    hit_label=[0,0,0,0]
    true_test=0
    predict_test = model.predict(test_data.T)
    if type(model) == MLPRegressor:
        predict_test = predict_test.argmax(1)
    for i in range(num_test):
        label[test_labels[i]] += 1
        if predict_test[i] == test_labels[i]:
            true_test += 1
            hit_label[test_labels[i]] += 1
    print("test acc:", true_test/num_test)
    print("anger prediction acc:",hit_label[0] / label[0])
    print("joy prediction acc:",hit_label[1] / label[1])
    print("optimism prediction acc:",hit_label[2] / label[2])
    print("sadness prediction acc:",hit_label[3] / label[3])
    return true_test/num_test
    