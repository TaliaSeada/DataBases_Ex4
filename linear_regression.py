import numpy as np
import matplotlib.pyplot as plt


class linear_regression:
    def __init__(self, data, learning_rate):
        self.data = data
        self.learning_rate = learning_rate
        self.W = np.zeros(len(data[0]) - 1)

    def gradients(self, x, y):
        h = x.dot(self.W)
        return np.mean(x.T * (h - y), axis=1)

    def loss(self, x, y):
        h = x.dot(self.W)
        return np.mean((h - y) ** 2) / 2

    def fit(self):
        Y = []
        X = self.data.copy()
        for i in X:
            Y.append(i[-1])
            del i[-1]
        X = np.array(X)
        Y = np.array(Y)

        ERR = 0
        while True:
            ERR_new = self.loss(X, Y)
            if abs(ERR - ERR_new) < 0.01:
                break
            ERR = ERR_new
            grad = self.gradients(X, Y)
            self.W = self.W - self.learning_rate * grad

        # print(self.W)

    def predict(self, test):
        Y = []
        X = test.copy()
        for i in X:
            Y.append(i[-1])
            del i[-1]
        X = np.array(X)
        Y = np.array(Y)

        h = X.dot(self.W)
        print("Predict: ", h)
        print("Actual: ", Y)

        plt.plot(Y)
        plt.plot(h)
        plt.show()
        return self.loss(X, Y)








