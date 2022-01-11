'''A tensorflow with version higher than 2.6 is expected
since in older versions, the MultiHeadAttention layer
does not exist.'''
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from data_process import load_text,load_label
from data_process import histogram_building
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence


# Reference: From the official document of Keras.


class TransformerLayer(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerLayer, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layerNorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layerNorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layerNorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layerNorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

def _transformer():
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




    embed_dim = 32  # Embedding size for each token
    num_heads = 4  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, num_feature, embed_dim)
    x = embedding_layer(inputs)
    transformer_layer = TransformerLayer(embed_dim, num_heads, ff_dim)
    x = transformer_layer(x)
    x = layers.GlobalAveragePooling1D()(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="tanh")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(4, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="RMSprop", loss="categorical_crossentropy", metrics=["accuracy"])
    model_fit = model.fit(train_seq_mat, Y_train, batch_size=64, epochs=15,
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
    _transformer()