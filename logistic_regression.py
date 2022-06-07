import numpy as np
import matplotlib.pyplot as plt


class logistic_regression:
    def __init__(self, data, learning_rate):
        self.data = data
        self.learning_rate = learning_rate
        self.W = np.zeros(len(data[0]) - 1)

    def gradients(self, x, y):
        h = 1 / (1 + np.exp(- self.W.dot(x.T)))
        return np.mean(x.T * (h - y), axis=1)

    def loss(self, x, y):
        h = 1 / (1 + np.exp(- self.W.dot(x.T)))
        return - np.mean(y * (np.log(h)) + (1 - y) * np.log(1 - h))

    def fit(self):
        Y = []
        X = self.data.copy()
        for i in X:
            Y.append(i[-2])
            del i[-2]
        X = np.array(X)
        Y = np.array(Y)

        ERR = 0
        while True:
            ERR_new = self.loss(X, Y)
            if abs(ERR - ERR_new) < 0.001:
                break
            ERR = ERR_new
            grad = self.gradients(X, Y)
            self.W = self.W - self.learning_rate * grad

        # print(self.W)

    def accuracy(self, h, y):
        cnt = 0
        length = len(h)
        for i in range(length):
            h[i] = round(h[i])
            if h[i] == y[i]:
                cnt += 1
        return cnt / length

    def recall(self, h, y):
        t_p = 0
        f_n = 0
        for i in range(len(h)):
            if h[i] == y[i] == 1:
                t_p += 1
            elif 0 == h[i] != y[i]:
                f_n += 1
        return t_p / (t_p + f_n)

    def precision(self, h, y):
        t_p = 0
        f_p = 0
        for i in range(len(h)):
            if h[i] == y[i] == 1:
                t_p += 1
            elif 1 == h[i] != y[i]:
                f_p += 1
        return t_p / (t_p + f_p)

    def f_score(self, recall, precision):
        if recall == 0 or precision == 0:
            return 0
        if recall == precision:
            return recall

        fScore = 2 * (precision * recall) / (precision + recall)
        return fScore

    def predict(self, test):
        Y = []
        X = test.copy()
        for i in X:
            Y.append(i[-2])
            del i[-2]
        X = np.array(X)
        Y = np.array(Y)

        h = 1 / (1 + np.exp(- self.W.dot(X.T)))
        accuracy = self.accuracy(h, Y)
        print("Accuracy = ", accuracy)

        recall = self.recall(h, Y)
        print("Recall = ", recall)

        precision = self.precision(h, Y)
        print("Precision = ", precision)

        fScore = self.f_score(recall, precision)
        print("F measure = ", fScore)

        print("Predict: ", h)
        print("Actual: ", Y)

        plt.plot(Y)
        plt.plot(h)
        plt.show()

