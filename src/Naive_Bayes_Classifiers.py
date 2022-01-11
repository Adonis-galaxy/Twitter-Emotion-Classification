from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import svm
import sklearn
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from base import baseline
from sklearn import svm
from sklearn.svm import SVC
from base import baseline
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.naive_bayes import ComplementNB  

class Naive_Bayes_Classifiers():
    def _knn(TF_ID):
        model = KNeighborsClassifier(n_neighbors=4)
        baseline(model,TF_ID)
    def _svm(TF_ID):
        model = svm.LinearSVC(penalty='l2', dual=False, max_iter=5, tol=1e-3,random_state=264, fit_intercept=True)
        val_acc=baseline(model,TF_ID)
    def _mlp(TF_ID):
        model = MLPRegressor(hidden_layer_sizes=(16),  activation='relu', solver='adam', max_iter=10)
        baseline(model,TF_ID)
    def _naive_nayes_mle(TF_ID):
        model = ComplementNB(alpha=1.1)
        baseline(model,TF_ID)
        