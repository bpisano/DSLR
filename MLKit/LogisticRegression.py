import MLKit
import math

class Logisticregression:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.thetas = {}
    
    def fit(self, data, row_names, feature_column_names):
        length = self.__length(data, feature_column_names)

        for feature_column_name in feature_column_names:
            self.thetas[feature_column_name] = {}
            for row_name in row_names:
                self.thetas[feature_column_name][row_name] = 0
                for _ in range(1000):
                    gradient_sum = 0
                    for g_row_name in row_names:
                        expected_result = 1 if g_row_name == row_name else 0
                        for value in data[g_row_name][feature_column_name]:
                            if value is None:
                                continue
                            
                            theta = self.thetas[feature_column_name][row_name]
                            gradient_sum += self.__gradient(value, expected_result, theta)
                    
                    self.thetas[feature_column_name][row_name] -= (gradient_sum / length[feature_column_name]) * self.learning_rate
    
    def save(self, file_name):
        MLKit.FileManager.save_model_data(self.thetas, file_name)
    
    @staticmethod
    def predict(x, theta):
        return Logisticregression.__h(x, theta)
    
    def __length(self, data, feature_column_names):
        length = {}
        
        for feature_name in feature_column_names:
            length[feature_name] = 0
            for row_name in data.keys():
                feature_data = data[row_name][feature_name]
                length[feature_name] += len(feature_data) - feature_data.count(None)
        
        return length
    
    def __gradient(self, x, y, theta):
        return (Logisticregression.__h(x, theta) - y) * x

    @staticmethod
    def __h(x, theta):
        return Logisticregression.__g(x * theta)

    @staticmethod
    def __g(z):
        return 1 / (1 + math.exp(-z))
