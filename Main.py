from linear_regression import linear_regression
from logistic_regression import logistic_regression

if __name__ == '__main__':
    print("LINEAR REGRESSION")
    # make the data
    data_txt = open("prices.txt")
    data = []
    for line in data_txt:
        line = line.split(",")
        for j in range(len(line)):
            line[j] = float(line[j])
        data.append(line)

    # linear regression
    linear_data = data.copy()
    # 75% for train
    linear_train = linear_data[:round(len(linear_data) * 0.75)]
    # 25% for test
    linear_test = linear_data[round(len(linear_data) * 0.75):]
    linear = linear_regression(linear_train, 0.0001)

    linear.fit()
    mse = linear.predict(linear_test)
    print("MSE = ", mse)

    print("\nLOGISTIC REGRESSION")
    # make the data
    data_txt = open("prices.txt")
    data = []
    for line in data_txt:
        line = line.split(",")
        for j in range(len(line)):
            line[j] = float(line[j])
        data.append(line)

    # logistic regression
    logistic_data = data.copy()
    # 85% for train
    logistic_train = logistic_data[:round(len(logistic_data) * 0.85)]
    # 15% for test
    logistic_test = logistic_data[round(len(logistic_data) * 0.85):]

    logistic = logistic_regression(logistic_train, 0.01)

    logistic.fit()
    logistic.predict(logistic_test)