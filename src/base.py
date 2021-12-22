# Created by xuyt1 on 2021/12/21
# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties
import numpy as np
from sklearn.neural_network import MLPRegressor
from text_preprocess import text_preprocessing
from label_preprocess import label_preprocessing
from train import trainFunc
from val import validation
from fileloader import load_text,load_label
def baseline(model = MLPRegressor( hidden_layer_sizes=(16),  activation='tanh', solver='adam')):
    # %%
    # load datasets
    train_text = load_text("train_text")
    train_labels = np.array(load_label("train_labels"))
    val_text = load_text("val_text")
    val_labels = np.array(load_label("val_labels"))
    test_text = load_text("test_text")
    test_labels = np.array(load_label("test_labels"))

    # %%

    from build_histogram import histogram_building
    histogram = histogram_building(train_text)
    num_feature = len(histogram) # 12887
    tokens = list(histogram.index)
    # %%
    train_data,num_train = text_preprocessing(train_text,tokens,num_feature)
    train_labels_processed = label_preprocessing(num_train,train_labels)
    val_data,num_val = text_preprocessing(val_text,tokens,num_feature)
    val_labels_processed = label_preprocessing(num_val,val_labels)

    # %%
    model = trainFunc(model,train_data,train_labels_processed,train_labels,num_train)
    validation(model,val_data,val_labels,num_val)