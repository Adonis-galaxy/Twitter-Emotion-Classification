from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  
from sklearn.naive_bayes import BernoulliNB  
from sklearn.naive_bayes import ComplementNB  
X, y = load_iris(return_X_y=True)
from base import baseline
gnb = ComplementNB(alpha=1.1)
baseline(gnb)

