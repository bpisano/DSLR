import MLKit

if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.predict("Hogwarts House", "houses_train.mlmodel")
    data_table.save("dataset/test_fill.csv")
