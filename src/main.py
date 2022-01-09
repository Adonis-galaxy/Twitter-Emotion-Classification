from sklearn.neural_network import MLPRegressor
import numpy as np
from sklearn import svm
import sklearn
from sklearn.svm import SVC
from sklearn import datasets       #导入数据模块
from sklearn.model_selection import train_test_split     #导入切分训练集、测试集模块
from sklearn.neighbors import KNeighborsClassifier
from base import baseline
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.svm import SVC
from base import baseline
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  
from sklearn.naive_bayes import BernoulliNB  
from sklearn.naive_bayes import ComplementNB  
from base import baseline
# model_names=['knn','svm','mlp','naive_bayes']

from knn import _knn
from svm import _svm
from mlp import _mlp
from naive_bayes import _naive_bayes

_knn()
_svm()
_mlp()
_naive_bayes()