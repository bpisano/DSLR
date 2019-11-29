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

    features = ["Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Ancient Runes", "Charms", "History of Magic"]
    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    data_table.train("Hogwarts House", features, "houses_train", learning_rate=0.001, accuracy_split=0.9)
