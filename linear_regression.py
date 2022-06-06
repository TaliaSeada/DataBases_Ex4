import numpy as np


class linear_regression:
    def __init__(self, data, learning_rate):
        self.data = data
        self.learning_rate = learning_rate
        self.W = np.zeros(len(data[0]) - 1)

    def gradients(self, x, y):
        return np.mean(x.T * (x.dot(self.W) - y), axis=1)

    def loss(self, x, y):
        return np.mean((x.dot(self.W) - y) ** 2) / 2

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
            if abs(ERR - ERR_new) < 0.001:
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

        print("Predict: ", X.dot(self.W))
        print("Actual: ", Y)
        return self.loss(X, Y)

if __name__ == '__main__':
    # make the data
    data_txt = open("prices.txt")
    data = []
    for line in data_txt:
        line = line.split(",")
        for j in range(len(line)):
            line[j] = float(line[j])
        data.append(line)

    # linear regression
    # 75% for train
    train = data[:int(len(data) * 0.75)]
    # 25% for test
    test = data[int(len(data) * 0.75):]
    lr = linear_regression(train, 0.0001)

    lr.fit()
    mse = lr.predict(test)
    print("MSE = ", mse)

    # logistic regression




