import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from data_process import load_text,load_label
from data_process import histogram_building
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

def _rnn_and_lstm(mode):
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
    max_len = 150
    tok = Tokenizer(num_words=max_words)
    Y_train = to_categorical(train_labels)
    Y_val = to_categorical(val_labels)
    Y_test = to_categorical(test_labels)

    tok.fit_on_texts(train_text)
    train_seq = tok.texts_to_sequences(train_text)
    val_seq = tok.texts_to_sequences(val_text)
    test_seq = tok.texts_to_sequences(test_text)
    train_seq_mat = sequence.pad_sequences(train_seq, maxlen=max_len)
    val_seq_mat = sequence.pad_sequences(val_seq, maxlen=max_len)
    test_seq_mat = sequence.pad_sequences(test_seq, maxlen=max_len)

    anger_test_list = []
    joy_test_list = []
    optimism_test_list = []
    sadness_test_list = []

    anger_test_labels = []
    joy_test_labels = []
    optimism_test_labels = []
    sadness_test_labels = []

    anger_Y_test = []
    joy_Y_test = []
    optimism_Y_test = []
    sadness_Y_test = []

    for i in range(len(test_labels)):
        if test_labels[i] == 0:
            anger_test_list.append(test_text[i])
            anger_Y_test.append(Y_test[i])
        elif test_labels[i] == 1:
            joy_test_list.append(test_text[i])
            joy_Y_test.append(Y_test[i])
        elif test_labels[i] == 2:
            optimism_test_list.append(test_text[i])
            optimism_Y_test.append(Y_test[i])
        elif test_labels[i] == 3:
            sadness_test_list.append(test_text[i])
            sadness_Y_test.append(Y_test[i])
        else:
            raise ValueError("Strange label!")

    anger_Y_test = np.array(anger_Y_test)
    joy_Y_test = np.array(joy_Y_test)
    optimism_Y_test = np.array(optimism_Y_test)
    sadness_Y_test = np.array(sadness_Y_test)

    anger_test_seq = tok.texts_to_sequences(anger_test_list)
    joy_test_seq = tok.texts_to_sequences(joy_test_list)
    optimism_test_seq = tok.texts_to_sequences(optimism_test_list)
    sadness_test_seq = tok.texts_to_sequences(sadness_test_list)

    anger_test_seq_mat = sequence.pad_sequences(anger_test_seq, maxlen=max_len)
    joy_test_seq_mat = sequence.pad_sequences(joy_test_seq, maxlen=max_len)
    optimism_test_seq_mat = sequence.pad_sequences(optimism_test_seq, maxlen=max_len)
    sadness_test_seq_mat = sequence.pad_sequences(sadness_test_seq, maxlen=max_len)

    '''SimpleRNN and LSTM are both implemented in this file.
    Change the name of that line leads to the switch.'''
    inputs = keras.layers.Input(name='inputs', shape=[max_len])
    layer = keras.layers.Embedding(max_words+1, 128, input_length=max_len)(inputs)
    if mode == 0:  # SimpleRNN Mode
        layer = keras.layers.SimpleRNN(128)(layer)  # For switching between SimpleRNN and LSTM
    elif mode == 1:   # LSTM Mode
        layer = keras.layers.LSTM(128)(layer)
    else:
        raise ValueError("Invalid mode!")
    layer = keras.layers.Dense(128, activation="relu")(layer)
    layer = keras.layers.Dropout(0.7)(layer)
    layer = keras.layers.Dense(4, activation="softmax")(layer)
    model = keras.Model(inputs=inputs, outputs=layer)

    # model.summary()
    model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

    '''The convergence of simple RNN is too slow that
    early stopping normally exits incorrectly.'''
    model_fit = model.fit(train_seq_mat, Y_train, batch_size=128, epochs=10,
                          validation_data=(val_seq_mat, Y_val),
                         )

    test_loss, test_acc = model.evaluate(test_seq_mat, Y_test)
    print("Test loss: ", test_loss, " Test acc: ", test_acc)

    anger_test_loss, anger_test_acc = model.evaluate(anger_test_seq_mat, anger_Y_test)
    print("ANGER Test loss: ", anger_test_loss, " ANGER Test acc: ", anger_test_acc)
    joy_test_loss, joy_test_acc = model.evaluate(joy_test_seq_mat, joy_Y_test)
    print("JOY Test loss: ", joy_test_loss, " JOY Test acc: ", joy_test_acc)
    optimism_test_loss, optimism_test_acc = model.evaluate(optimism_test_seq_mat, optimism_Y_test)
    print("OPTIMISM Test loss: ", optimism_test_loss, " OPTIMISM Test acc: ", optimism_test_acc)
    sadness_test_loss, sadness_test_acc = model.evaluate(sadness_test_seq_mat, sadness_Y_test)
    print("SADNESS Test loss: ", sadness_test_loss, " SADNESS Test acc: ", sadness_test_acc)

if __name__ == "__main__":
    mode = 0
    _rnn_and_lstm(mode)
