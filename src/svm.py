from sklearn import svm
from sklearn.svm import SVC
from base import baseline
import numpy as np
def _svm():
    model = svm.LinearSVC(penalty='l1',dual=False,max_iter=4,tol=1e-3,random_state=264)
    val_acc=baseline(model)

if __name__ == '__main__':
    _svm()