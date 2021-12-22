from sklearn import svm
from sklearn.svm import SVC
from base import baseline
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# model = svm.LinearSVC()
model = Pipeline([
                                ("scaler", StandardScaler()), 
                                ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001))
                            ])

baseline(model)

#
