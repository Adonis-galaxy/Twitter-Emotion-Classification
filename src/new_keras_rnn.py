import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from data_process.fileloader import load_text,load_label
from data_process.build_histogram import histogram_building
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

train_text = load_text("train_text")
train_labels = np.array(load_label("train_labels"))
val_text = load_text("val_text")
val_labels = np.array(load_label("val_labels"))
test_text = load_text("test_text")
test_labels = np.array(load_label("test_labels"))
histogram = histogram_building(train_text)
num_feature = len(histogram)  # 12887
tokens = list(histogram.index)


'''Map the text into word vectors and 
then pad the sequences to the same length 
for further processing'''
max_words = 12887
max_len = 200
tok = Tokenizer(num_words=max_words)
Y_train = to_categorical(train_labels)
Y_val = to_categorical(val_labels)
Y_test = to_categorical(test_labels)
tok.fit_on_texts(train_text)
train_seq = tok.texts_to_sequences(train_text)
val_seq = tok.texts_to_sequences(val_text)
test_seq = tok.texts_to_sequences(test_text)
train_seq_mat = sequence.pad_sequences(train_seq,maxlen=max_len)
val_seq_mat = sequence.pad_sequences(val_seq,maxlen=max_len)
test_seq_mat = sequence.pad_sequences(test_seq,maxlen=max_len)

'''SimpleRNN and LSTM are both implemented in this file. 
Change the name of that line leads to the switch.'''
inputs = keras.layers.Input(name='inputs',shape=[max_len])
layer = keras.layers.Embedding(max_words+1,128,input_length=max_len)(inputs)
layer = keras.layers.SimpleRNN(128)(layer) #For switching between SimpleRNN and LSTM
layer = keras.layers.Dense(128,activation="relu",name="FC1")(layer)
layer = keras.layers.Dropout(0.5)(layer)
layer = keras.layers.Dense(4,activation="softmax",name="FC2")(layer)
model = keras.Model(inputs=inputs,outputs=layer)
model.summary()
model.compile(loss="categorical_crossentropy",optimizer=keras.optimizers.RMSprop(),metrics=["accuracy"])

'''The convergence of simple RNN is too slow that 
early stopping normally exits incorrectly.'''
model_fit = model.fit(train_seq_mat,Y_train,batch_size=128,epochs=10,
                      validation_data=(val_seq_mat,Y_val),
                      # callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0.00001)]
                     )

test_loss, test_acc = model.evaluate(test_seq_mat, Y_test)
print(test_loss, test_acc)

