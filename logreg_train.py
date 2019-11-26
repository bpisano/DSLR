import MLKit
import copy
import math


def dict_with_features(features):
    dct = dict()
    for feature in features:
        dct[feature] = 0
    return dct


if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()

    features = ["Astronomy", "Herbology", "Divination", "Muggle Studies",
                "Ancient Runes", "History of Magic", "Transfiguration", "Charms", "Flying"]
    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    features_for_labels = dict(
        Ravenclaw=dict_with_features(("theta0", "Astronomy", "Herbology", "Muggle Studies", "Ancient Runes", "Charms")),
        Slytherin=dict_with_features(("theta0", "Astronomy", "Herbology", "Divination", "Ancient Runes")),
        Gryffindor=dict_with_features(("theta0", "Astronomy", "Herbology", "Divination", "Flying", "Transfiguration",
                                       "History of Magic")),
        Hufflepuff=dict_with_features(("theta0", "Astronomy", "Herbology", "Ancient Runes")))
    total_len = 0
    data = data_table.feature_values_for_rows_in_target_column("Hogwarts House", houses, features)
    for house in data.keys():
        total_len += len(data[house])
    for house in houses:
        for _ in range(500):
            gradient_sum = [0] * len(features_for_labels[house].keys())
            for house_to_test in houses:
                expected_result = 1 if house_to_test == house else 0
                for student in data[house_to_test]:
                    sum = features_for_labels[house]["theta0"]
                    for feature in features_for_labels[house].keys():
                        if feature == "theta0":
                            continue
                        sum += student[feature] * features_for_labels[house][feature]
                    base_result = ((1 / 1 + math.exp(sum)) - expected_result)
                    gradient_sum[0] += base_result
                    for index, feature in enumerate(features_for_labels[house].keys()):
                        if feature == "theta0":
                            continue
                        gradient_sum[index] += base_result * student[feature]
            for index, feature in enumerate(features_for_labels[house].keys()):
                features_for_labels[house][feature] -= 0.00001 * gradient_sum[index] / total_len
    MLKit.FileManager.save_model_data(features_for_labels, "houses_train")
    # for label in features_for_labels.keys():
    #     self.thetas[label] = {}
    #     for feature in features_for_labels[label]:
    #         self.thetas[label][feature] = 0
    #         for _ in range(100):
    #             gradient_sum = 0
    #             for g_row_name in row_names:
    #                 expected_result = 1 if g_row_name == row_name else 0
    #                 for value in data[g_row_name][feature_column_name]:
    #                     if value is None:
    #                         continue
    #
    #                     theta = self.thetas[feature_column_name][row_name]
    #                     gradient_sum += self.__gradient(value, expected_result, theta)
    #
    #             self.thetas[feature_column_name][row_name] -= (gradient_sum / length[
    #                 feature_column_name]) * self.learning_rate

