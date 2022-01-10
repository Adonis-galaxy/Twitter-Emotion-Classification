# model_names=['knn','svm','mlp','naive_bayes']
from knn import _knn
from svm import _svm
from mlp import _mlp
from naive_bayes import _naive_bayes
from new_keras_rnn import _rnn_and_lstm
from transformer import _transformer
TF_ID = True
_knn(TF_ID)
_svm(TF_ID)
_mlp(TF_ID)
_naive_bayes(TF_ID)

_rnn_and_lstm(0)  #rnn
_rnn_and_lstm(1)  #lstm
_transformer()
