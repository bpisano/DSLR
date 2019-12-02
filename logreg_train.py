import MLKit
import math

if __name__ == "__main__":
    # Features "Astronomy" "Herbology" "Defense Against the Dark Arts" "Divination" "Charms" "History of Magic"
    # Target column "Hogwarts House"
    MLKit.CommandLine.register_flag("X", description="The features to train.", has_multiple_values=True)
    MLKit.CommandLine.register_flag("Y", description="The column to train.")
    MLKit.CommandLine.register_flag("o", description="The output file name.", default_value="train")
    MLKit.CommandLine.register_flag("l", description="The learning rate.", default_value=0.0001)
    MLKit.CommandLine.register_flag("a", description="The accuracy split to train the data.")
    MLKit.CommandLine.register_usage("Build a model from a csv file.\nlogreg_train.py [train_model]")
    MLKit.CommandLine.show_usage_if_needed()

    input_file_name = MLKit.CommandLine.get_argument_at_index(1)
    output_file_name = MLKit.CommandLine.get_value_for_flag("o")
    target_column = MLKit.CommandLine.get_value_for_flag("Y")
    features = MLKit.CommandLine.get_value_for_flag("X")
    learning_rate = float(MLKit.CommandLine.get_value_for_flag("l"))
    accuracy_split = float(MLKit.CommandLine.get_value_for_flag("a"))

    data_table = MLKit.DataTable(input_file_name)
    data_table.train(target_column, features, output_file_name, learning_rate=learning_rate, accuracy_split=accuracy_split)
