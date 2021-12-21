# Created by xuyt1 on 2021/12/21
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.neural_network import MLPRegressor

# %%
# dataloader
train_text=[]
with open("../data/train_text.txt") as f:
    for line in f:
        train_text.append(line.strip('\n'))
train_labels=[]
with open("../data/train_labels.txt") as f:
    for line in f:
        train_labels.append(int(line.strip('\n')))
val_text=[]
with open("../data/val_text.txt") as f:
    for line in f:
        val_text.append(line.strip('\n'))
val_labels=[]
with open("../data/val_labels.txt") as f:
    for line in f:
        val_labels.append(int(line.strip('\n')))

# %%
histogram = {}
for sentence in train_text:
    words = sentence.split() # tokenlize
    for word in words:
        if word not in histogram.keys():
            histogram[word]=1
        else:
            histogram[word] += 1
histogram = pd.Series(histogram)
num_feature = len(histogram) # 12887


num_train = len(train_text)
train_data = np.zeros(shape=(num_feature, num_train)) # shape (12887, 3257)

tokens = list(histogram.index)
for i in range(num_train):
    for word in train_text[i].split():
        index = tokens.index(word)
        train_data[index,i]+=1

old_train_labels = train_labels.copy()
 
train_labels = np.zeros(shape=(num_train,4))
for i in range(num_train):
    train_labels[i,old_train_labels[i]]=1
 

model_mlp=MLPRegressor( hidden_layer_sizes=(16),  activation='tanh', solver='adam')
model_mlp.fit(train_data.T, train_labels)


 
train_accuracy = sum(model_mlp.predict(train_data.T).argmax(1) == old_train_labels) / num_train
print("train acc",train_accuracy)

 
# val


num_val = len(val_text)
val_data = np.zeros(shape=(num_feature, num_val))
# print(train_data.shape) # (?, 3257)
tokens = list(histogram.index)
for i in range(num_val):
    for word in val_text[i].split():
        try:
            index = tokens.index(word)
            val_data[index,i]+=1
        except ValueError: # some data in val may not appeat in training set
            pass

old_val_labels = val_labels.copy()
val_labels = np.zeros(shape=(num_train,4))
for i in range(num_val):
    val_labels[i,old_val_labels[i]]=1
 
old_val_labels = np.array(old_val_labels)
true_val=0
predict_val = model_mlp.predict(val_data.T).argmax(1)
for i in range(num_val):
    if predict_val[i] == old_val_labels[i]:
        true_val += 1
print("val acc:", true_val/num_val)

