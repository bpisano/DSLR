import MLKit
import math
import numpy as np


class LogisticRegression:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.thetas = {}

    def fit(self, target_column, feature_columns):
        X = np.asarray([column.values for column in feature_columns], dtype="float64")
        X = X.T
        Y = np.asarray(target_column.values)
        X = np.array([val for val in X if not np.isnan(val).any()])
        X = X.T
        thetas = np.zeros((np.unique(Y).shape[0], X.shape[0]))
        m = np.shape(Y)[0]
        linear_matrix = np.ones(m)
        for index, row_name in enumerate(np.unique(Y)):
            expected_results = np.where(Y == row_name, 1, 0)
            row_theta = thetas[index]
            prev_cost = -100
            cost = 0
            while abs(cost - prev_cost) > 0.000001:
                x = LogisticRegression.__g(row_theta.dot(X))
                x[x == 1] = 0.999
                prev_cost = cost
                cost = np.log(x) * expected_results + (1 - expected_results) * np.log(1 - x)
                cost = cost.dot(linear_matrix) / -m
                error = x - expected_results
                gradient = error * X
                new_thetas = linear_matrix.dot(gradient.T)
                new_thetas = new_thetas * self.learning_rate / m
                thetas[index] -= new_thetas
        thetas_dict = dict()
        feature_names = [column.name for column in feature_columns]
        for index1, row_name in enumerate(np.unique(Y)):
            thetas_dict[row_name] = dict()
            for index2, feature in enumerate(feature_names):
                thetas_dict[row_name][feature] = float(thetas[index1][index2])
        MLKit.FileManager.save_model_data(thetas_dict, "houses_train")

    def save(self, file_name):
        MLKit.FileManager.save_model_data(self.thetas, file_name)

    @staticmethod
    def predict(x):
        return LogisticRegression.__g(x)

    def __length(self, data, feature_column_names):
        length = {}

        for feature_name in feature_column_names:
            length[feature_name] = 0
            for row_name in data.keys():
                feature_data = data[row_name][feature_name]
                length[feature_name] += len(feature_data) - feature_data.count(None)

        return length

    def __gradient(self, x, y, theta):
        return (LogisticRegression.__h(x, theta) - y) * x

    @staticmethod
    def __h(x, theta):
        return LogisticRegression.__g(x * theta)

    @staticmethod
    def __g(z):
        return 1 / (1 + np.exp(-z))
