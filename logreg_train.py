import MLKit

if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()

    houses = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    features = ["Divination", "Muggle Studies", "History of Magic", "Transfiguration", "Charms", "Flying"]

    data_table.train("Hogwarts House", houses, features, "houses_train", learning_rate=0.0001)
