from sklearn.neural_network import MLPRegressor
from base import baseline
def _mlp():
    model = MLPRegressor(hidden_layer_sizes=(16),  activation='relu', solver='adam', max_iter=10)
    baseline(model)

if __name__ == '__main__':
    _mlp()
