from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import svm
import sklearn
from sklearn.svm import SVC
def trainFunc(model,train_data,train_labels_processed,train_labels,num_train):
    # model=MLPRegressor( hidden_layer_sizes=(16),  activation='tanh', solver='adam')
    print("type of the model:",type(model))
    if type(model) == MLPRegressor:
        model.fit(train_data.T, train_labels_processed)
        train_accuracy = sum(model.predict(train_data.T).argmax(1) == train_labels) / num_train
        print("train acc",train_accuracy)
        return model
    elif type(model) == sklearn.pipeline.Pipeline or svm.LinearSVC: # svm
        model.fit(train_data.T, train_labels)
        train_accuracy = sum(model.predict(train_data.T) == train_labels) / num_train
        print("train acc",train_accuracy)
        return model
    else:
        raise RuntimeError("What model? Write your model train function here")
    