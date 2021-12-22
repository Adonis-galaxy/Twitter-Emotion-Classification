import numpy as np
def text_preprocessing(text,tokens,num_feature):
    num_data = len(text)
    data = np.zeros(shape=(num_feature, num_data))
    # print(train_data.shape) # (?, 3257)
    for i in range(num_data):
        for word in text[i].split():
            try:
                index = tokens.index(word)
                data[index,i]+=1
            except ValueError: # some data in val may not appeat in training set
                pass
    return data,num_data