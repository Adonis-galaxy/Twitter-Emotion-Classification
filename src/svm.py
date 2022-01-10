from sklearn import svm
from sklearn.svm import SVC
from base import baseline
import numpy as np

def _svm():
    model = svm.LinearSVC(penalty='l2', dual=False, max_iter=5, tol=1e-3,random_state=264, fit_intercept=True)
    val_acc=baseline(model)

if __name__ == '__main__':
    _svm()
