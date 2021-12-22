import pandas as pd
def histogram_building(text):
    histogram = {}
    for sentence in text:
        words = sentence.split() # tokenlize
        for word in words:
            if word not in histogram.keys():
                histogram[word]=1
            else:
                histogram[word] += 1
    histogram = pd.Series(histogram)
    return histogram
