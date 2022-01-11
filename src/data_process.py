import pandas as pd
def histogram_building(text, bag=1):
    histogram = {}
    for i in range(bag):
        for sentence in text:
            words = sentence.split() # tokenlize
            for j in range(len(words) - i):
                temp = ''
                for k in range(i + 1):
                    temp += words[j]
                if temp not in histogram.keys():
                    histogram[temp]=1
                else:
                    histogram[temp] += 1
    histogram = pd.Series(histogram)
    return histogram
def load_text(file_name):
    lst=[]
    with open("../data/"+file_name+".txt", encoding='utf8') as f:
        for line in f:
            lst.append(line.strip('\n'))
    return lst
def load_label(file_name):
    lst=[]
    with open("../data/"+file_name+".txt", errors='ignore') as f:
        for line in f:
            lst.append(int(line.strip('\n')))
    return lst
import numpy as np
def label_preprocessing(num_train,train_labels):
    train_labels_processed = np.zeros(shape=(num_train,4))
    for i in range(num_train):
        train_labels_processed[i,int(train_labels[i])]=1
    return train_labels_processed
"""
before preprocess:[0,2,3,1]
after preprocess:
[
    [1,0,0,0]
    [0,0,1,0]
    [0,0,0,1]
    [0,1,0,0]
]
"""
import numpy as np
def text_preprocessing(text,tokens,num_feature, bag=1):
    num_data = len(text)
    data = np.zeros(shape=(num_feature, num_data))
    # print(train_data.shape) # (?, 3257)
    for l in range(bag):
        for i in range(num_data):
            words = text[i].split()
            for j in range(len(words) - l):
                temp = ''
                for k in range(l + 1):
                    temp += words[j]
                try:
                    if temp in tokens:
                        index = tokens.index(temp)
                        data[index, i] += 1
                except ValueError: # some data in val may not appear in training set
                    pass
    return data, num_data