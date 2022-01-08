from sklearn.ensemble import RandomForestClassifier
from base import baseline
from sklearn.naive_bayes import ComplementNB  
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
# clf = tree.DecisionTreeClassifier()
print(type(clf))
baseline(clf)
# clf.fit(train_x, train_y)