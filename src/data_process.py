import pandas as pd
import re
import numpy as np


def data_loader(text):
    train_text = text
    train_data = []
    l = 0
    for sentence in train_text:
        words = re.split(r'[“:;,.!#?*()\s]\s*', sentence)
        train_data.append([])
        for word in words:
            if word == '':
                continue
            start = 0
            for i in range(len(word)):
                if i == 0 and word[i] == '\'':
                    start = start + 1
                    continue
                if i == len(word) - 1 and word[i] == '\'':
                    word = word[:-1]
                    break
                if word[i:i+2] == '\\n':
                    if start != i:
                        train_data[l].append(word[start:i])
                    start = i + 2
                    i = i + 1
                if ord(word[i]) > 10000:
                    if word[i] != '❤':
                        if start != i:
                            train_data[l].append(word[start:i])
                        train_data[l].append(word[i])
                        start = i + 1
                    else:
                        if start != i:
                            train_data[l].append(word[start:i])
                        train_data[l].append(word[i:i+2])
                        start = i + 2
                        i = i + 1
            if start != len(word):
                train_data[l].append(word[start:len(word)])
        l += 1
    return train_data


def histogram_building(text, bag=1):
    histogram = {}
    for i in range(bag):
        for words in text:
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
    data = data_loader(lst)
    return data


def load_label(file_name):
    lst=[]
    with open("../data/"+file_name+".txt", errors='ignore') as f:
        for line in f:
            lst.append(int(line.strip('\n')))
    return lst



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

def text_preprocessing(text,tokens,num_feature, bag=1):
    num_data = len(text)
    data = np.zeros(shape=(num_feature, num_data))
    # print(train_data.shape) # (?, 3257)
    for l in range(bag):
        for i in range(num_data):
            words = text[i]
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
