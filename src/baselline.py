# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.neural_network import MLPRegressor
# %%
# dataloader
def dataloader(para):
    lst=[]
    with open("../data/"+para+".txt") as f:
        for line in f:
            lst.append(line.strip('\n'))
    return lst
train_text = dataloader("train_text")
train_labels = np.array(dataloader("train_labels"))
val_text = dataloader("val_text")
val_labels = np.array(dataloader("val_labels"))
test_text = dataloader("test_text")
test_labels = np.array(dataloader("test_labels"))


# %%
# histogram
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

# %% text process

