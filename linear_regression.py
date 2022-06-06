import numpy as np
class linear_regression:
    def __init__(self, data, learning_rate):
        self.data = data
        self.learning_rate = learning_rate
        self.m = 0
        self.b = 0

    def fit(self):
        print()
