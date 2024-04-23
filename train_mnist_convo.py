from keras.datasets import mnist
from keras.utils import to_categorical

from NN_network import *
from NN_layers import *

import numpy as np


def preprocess_data(x, y, limit):
    x = x.reshape(len(x), 1, 28, 28)
    x[x > 0] = 1

    y = to_categorical(y)
    y = y.reshape(len(y), 10, 1)

    return x[:limit], y[:limit]

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 60000)
x_test, y_test = preprocess_data(x_test, y_test, 10000)

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

# train(network, mse, mse_prime, 100, 0.1, x_train, y_train)

# save_model(network, "mnist_model_conv_black_white")
load_model(network, "mnist_model_conv_black_white")


numbers = [["zero", 0],["one", 0],["two", 0],["three", 0],["four", 0],["five", 0],["six", 0],["seven", 0],["eight", 0],["nine", 0],]

right = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    if (np.argmax(output) == np.argmax(y)):
        right += 1
    else:
        numbers[np.argmax(y)][1] += 1;
    

print("")
print("model: {}/{}, acc: {}%".format(right, x_test.shape[0], (100 / x_test.shape[0]) * right))
for i in numbers:
    print("{}:\t{}".format(i[0], i[1]))