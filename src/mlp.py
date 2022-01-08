from sklearn.neural_network import MLPRegressor
from base import baseline
model = model=MLPRegressor( hidden_layer_sizes=(16),  activation='relu', solver='adam')
baseline(model)