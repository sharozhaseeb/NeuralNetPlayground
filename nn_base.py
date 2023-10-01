import numpy as np

class NeuralNetwork:
    def __init__(self, layer_dims):
        self.layer_dims = layer_dims
        self.parameters = self.initialize_parameters()
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def initialize_parameters(self):
        np.random.seed(0)
        parameters = {}
        L = len(self.layer_dims)  # Number of layers

        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((self.layer_dims[l], 1))

        return parameters

    def forward_pass(self, X):
        caches = []
        A = X
        L = len(self.parameters) // 2  # Number of layers

        for l in range(1, L):
            A_prev = A
            Z = np.dot(self.parameters['W' + str(l)], A_prev) + self.parameters['b' + str(l)]
            A = self.relu(Z)
            cache = (A_prev, self.parameters['W' + str(l)], self.parameters['b' + str(l)], Z)
            caches.append(cache)

        ZL = np.dot(self.parameters['W' + str(L)], A) + self.parameters['b' + str(L)]
        AL = self.sigmoid(ZL)
        cache = (A, self.parameters['W' + str(L)], self.parameters['b' + str(L)], ZL)
        caches.append(cache)

        return AL, caches

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL)) / m
        return cost

    def backward_pass(self, AL, Y, caches):
        grads = {}
        L = len(caches)  # Number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)  # Ensure the same shape as AL

        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L - 1]
        grads['dA' + str(L - 1)], grads['dW' + str(L)], grads['db' + str(L)], grads['dZ' + str(L)] = self.linear_activation_backward(
            dAL, current_cache, activation='sigmoid')

        for l in reversed(range(L - 1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp, dZ_temp = self.linear_activation_backward(
                grads['dA' + str(l + 1)], current_cache, activation='relu')
            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l + 1)] = dW_temp
            grads['db' + str(l + 1)] = db_temp
            grads['dZ' + str(l + 1)] = dZ_temp

        return grads

    def linear_activation_backward(self, dA, cache, activation):
        A_prev, W, b, Z = cache
        m = A_prev.shape[1]

        if activation == "relu":
            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(W.T, dZ)

        elif activation == "sigmoid":
            s = self.sigmoid(Z)
            dZ = dA * s * (1 - s)
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db, dZ

    def update_parameters(self, grads, learning_rate):
        L = len(self.parameters) // 2  # Number of layers

        for l in range(1, L + 1):
            self.parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
            self.parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    def fit(self, X, Y, learning_rate, epochs):
        for epoch in range(epochs):
            AL, caches = self.forward_pass(X)
            cost = self.compute_cost(AL, Y)
            grads = self.backward_pass(AL, Y, caches)
            self.update_parameters(grads, learning_rate)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Cost = {cost}")

        return self.parameters