import MLKit

if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()

    target_column = "Hogwarts House"
    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    features = ["Divination", "Muggle Studies", "History of Magic", "Transfiguration", "Charms", "Flying"]

    print(data_table.feature_values_for_rows_in_target_column(target_column, houses, ["Astronomy", "Charms"]))

    # data_table.train("Hogwarts House", houses, features, "houses_train", learning_rate=0.0001)
