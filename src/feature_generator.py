from data_process import load_text, histogram_building
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import re


def feature_extractor():
    data_text = load_text("train_text")
    histogram = histogram_building(data_text)
    old_feature = list(histogram.index)
    num = [0 for _ in range(len(old_feature))]
    pass


feature_extractor()
