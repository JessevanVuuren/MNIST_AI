from keras.utils import to_categorical
from keras.datasets import mnist
from NN_network import *
from NN_layers import *

import numpy as np

def preprocess_data(x, y, limit):
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255

    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 60000)
x_test, y_test = preprocess_data(x_test, y_test, 50)

network = [
    Dense(28 * 28, 128),
    Sigmoid(),
    Dense(128, 64),
    Sigmoid(),
    Dense(64, 10),
    Sigmoid()
]


# train(network, mse, mse_prime, 30, 0.1, x_train, y_train)

# save_model(network, "mnist_model_dense")
# load_model(network, "mnist_model_dense")

for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('pred:', np.argmax(output), ' - true:', np.argmax(y), np.argmax(output) == np.argmax(y))


