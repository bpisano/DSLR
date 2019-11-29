import MLKit
import math
import numpy as np

class LogisticRegression:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.thetas = {}
    
    def fit(self, X, Y):
        thetas = np.zeros((np.unique(Y).shape[0], X.shape[0]))
        m = np.shape(Y)[0]
        linear_matrice = np.ones(m)

        # print("Shape X: ", X.shape)
        # print("Shape thetas: ", thetas.shape)

        # print(thetas)
        # print(x)
        # print(X.shape, thetas.shape, thetas.dot(X).shape)
        # np.place(y == None, 0, y)

        for index, row_name in enumerate(np.unique(Y)):
            print(row_name)
            expected_results = np.where(Y == row_name, 1, 0)
            row_theta = thetas[index]
            # print(expected_results)
            # print(expected_results.shape)

            # print(row_name)
            for _ in range(10000):
                x = row_theta.dot(X)
                error = LogisticRegression.__g(x) - expected_results
                gradient = error * X

                # print("Shape error: ", error.shape)
                # print("Shape gradient: ", gradient.shape)
                # print(gradient)
                
                new_thetas = linear_matrice.dot(gradient.T)
                new_thetas = new_thetas * self.learning_rate / m
                thetas[index] -= new_thetas

                # print("Shape new_thetas: ", new_thetas.shape)
            # break
            # print("\n")
        
        print(thetas)
                # x_matrice = x.dot(unity_matrice)
                # print(x_matrice)
                # print(x.shape, unity_matrice.shape, x_matrice.shape)
                # error = LogisticRegression.__g(x_matrice) - expected_results
                # print(output)


        # print(y)
        # length = self.__length(data, feature_column_names)

        # for feature_column_name in feature_column_names:
        #     self.thetas[feature_column_name] = {}
        #     for row_name in row_names:
        #         self.thetas[feature_column_name][row_name] = 0
        #         for _ in range(1000):
        #             gradient_sum = 0
        #             for g_row_name in row_names:
        #                 expected_result = 1 if g_row_name == row_name else 0
        #                 for value in data[g_row_name][feature_column_name]:
        #                     if value is None:
        #                         continue
                            
        #                     theta = self.thetas[feature_column_name][row_name]
        #                     gradient_sum += self.__gradient(value, expected_result, theta)
                    
        #             self.thetas[feature_column_name][row_name] -= (gradient_sum / length[feature_column_name]) * self.learning_rate
    
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
