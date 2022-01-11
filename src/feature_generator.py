from data_process import load_text, histogram_building
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re


def feature_extractor():
    data_text = load_text("train_text")
    histogram = histogram_building(data_text)
    old_feature = list(histogram.index)
    print(old_feature)


feature_extractor()
