from sklearn.neural_network import MLPRegressor
from base import baseline
model = MLPRegressor( hidden_layer_sizes=(16),  activation='relu', solver='adam',max_iter=10)
baseline(model)