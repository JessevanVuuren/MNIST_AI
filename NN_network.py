from NN_layers import Dense, Convolutional
import pickle

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)

    return output

def train(network, loss, loss_prime, epoch, learning_rate, X, Y):
    
    for i in range(epoch):
        error = 0
        
        for x, y in zip(X, Y):

            output = predict(network, x)

            error += loss(y, output)

            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        
        error /= len(X)
        print("{}/{}, error={}".format(i + 1, epoch, error))


def save_model(network, name):
    with open(name + ".pkl", "wb") as file:
        model = {}
        count = 0
        for i in range(len(network)):
            if (type(network[i]) == Dense):
                count += 1
            
                weights = "layer_{}_weights".format(count)
                bias = "layer_{}_bias".format(count)
                                
                model[weights] = network[i].weights
                model[bias] = network[i].bias

            if (type(network[i]) == Convolutional):
                count += 1
            
                kernels = "layer_{}_kernels".format(count)
                biases = "layer_{}_biases".format(count)
                                
                model[kernels] = network[i].kernels
                model[biases] = network[i].biases


        pickle.dump(model, file)

        
def load_model(network, name):
    with open(name + ".pkl", "rb") as file:
        model = pickle.load(file)
        count = 0
        for i in range(len(network)):
            if (type(network[i]) == Dense):
                count += 1
            
                weights = "layer_{}_weights".format(count)
                bias = "layer_{}_bias".format(count)

                network[i].weights = model[weights]
                network[i].bias = model[bias]

            if (type(network[i]) == Convolutional):
                count += 1
            
                kernels = "layer_{}_kernels".format(count)
                biases = "layer_{}_biases".format(count)
                                
                network[i].kernels = model[kernels]
                network[i].biases = model[biases]
                