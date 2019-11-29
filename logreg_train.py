import MLKit
import math


def dict_with_features(features_use):
    dct = {}
    for feature_use in features_use:
        dct[feature_use] = 0
    return dct


if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()

    features = ["Astronomy", "Herbology", "Defense Against the Dark Arts", "Ancient Runes"]
    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    data_table.train("Hogwarts House", features, "test_fill.csv", learning_rate=0.000001)
    # features_for_labels = dict(
    #     Ravenclaw=dict_with_features(("theta0", "Astronomy", "Herbology", "Muggle Studies", "Ancient Runes", "Charms")),
    #     Slytherin=dict_with_features(("theta0", "Astronomy", "Herbology", "Divination", "Ancient Runes")),
    #     Gryffindor=dict_with_features(("theta0", "Astronomy", "Herbology", "Divination", "Flying", "Transfiguration",
    #                                    "History of Magic")),
    #     Hufflepuff=dict_with_features(("theta0", "Astronomy", "Herbology", "Ancient Runes")))
    # total_len = 0
    # data = data_table.feature_values_for_rows_in_target_column("Hogwarts House", houses, features)
    # for house in data.keys():
    #     total_len += len(data[house])
    # for house in houses:
    #     previous_cost = -float("inf")
    #     cost = 0
    #     while abs(cost - previous_cost) > 0.1:
    #         previous_cost = cost
    #         cost = 0
    #         gradient_sum = [0] * len(features_for_labels[house].keys())
    #         for house_to_test in houses:
    #             expected_result = 1 if house_to_test == house else 0
    #             for student in data[house_to_test]:
    #                 sum_thetas_by_marks = features_for_labels[house]["theta0"]
    #                 for feature in features_for_labels[house].keys():
    #                     if feature == "theta0":
    #                         continue
    #                     sum_thetas_by_marks += student[feature] * features_for_labels[house][feature]
    #                 base_result = (1 / (1 + math.exp(-sum_thetas_by_marks)))
    #                 base_result = 0.999 if base_result == 1 else base_result
    #                 cost += expected_result * math.log(base_result) + ((1 - expected_result) * math.log(1 - base_result))
    #                 gradient_sum[0] += base_result - expected_result
    #                 for index, feature in enumerate(features_for_labels[house].keys()):
    #                     if feature == "theta0":
    #                         continue
    #                     gradient_sum[index] += (base_result - expected_result) * student[feature]
    #         cost /= total_len
    #         for index, feature in enumerate(features_for_labels[house].keys()):
    #             features_for_labels[house][feature] -= 0.001 * gradient_sum[index] / total_len
    # print(features_for_labels)
    # MLKit.FileManager.save_model_data(features_for_labels, "houses_train")
