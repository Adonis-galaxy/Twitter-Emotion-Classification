# Created by xuyt1 on 2021/12/21
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.neural_network import MLPRegressor
import sys

from data_process.text_preprocess import text_preprocessing
from data_process.label_preprocess import label_preprocessing
from train import trainFunc
from val import validation
from data_process.fileloader import load_text,load_label
from data_process.build_histogram import histogram_building
from sklearn.svm import LinearSVC

def baseline(model = LinearSVC()):
    # %%
    # load datasets
    train_text = load_text("train_text")
    train_labels = np.array(load_label("train_labels"))
    val_text = load_text("val_text")
    val_labels = np.array(load_label("val_labels"))
    test_text = load_text("test_text")
    test_labels = np.array(load_label("test_labels"))
    # %%
    histogram = histogram_building(train_text)
    num_feature = len(histogram) # 12887
    tokens = list(histogram.index)
    # %%
    train_data,num_train = text_preprocessing(train_text,tokens,num_feature)
    val_data,num_val = text_preprocessing(val_text,tokens,num_feature)
    # %%
    trainFunc(model,train_data,train_labels,num_train)
    validation(model,val_data,val_labels,num_val)
# %%
if __name__ == '__main__':
    baseline()