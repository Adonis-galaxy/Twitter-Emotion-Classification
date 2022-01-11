from data_process import load_text, load_label
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re


def train_data_loader():
    lst = []
    with open("../data/" + "train_text" + ".txt", encoding='utf8') as f:
        for line in f:
            lst.append(line.strip('\n'))
    train_text = lst
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


def feature_extractor():
    pass


train_data_loader()

