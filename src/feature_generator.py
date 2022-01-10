import sklearn
import numpy as np
from data_process.text_preprocess import text_preprocessing
from data_process.label_preprocess import label_preprocessing
from data_process.fileloader import load_text, load_label
from data_process.build_histogram import histogram_building
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

def feature_set():
    train_text = load_text("train_text")
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(train_text)
    dic = vectorizer.get_feature_names()
    transformer = TfidfTransformer()
    tf_idf = transformer.fit_transform(X)
    feature = []
    for l in tf_idf:
        indice = l.indices
        data = l.data
        temp = []
        for i in range(len(data)):
            temp.append((indice[i], data[i]))
        temp.sort(key=lambda x: -x[1])
        for i in range(min(len(temp), 3)):
            word = dic[temp[i][0]]
            if word not in feature:
                feature.append(word)
    return feature

