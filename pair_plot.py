import MLKit

if __name__ == "__main__":
    # Features "Astronomy" "Herbology" "Defense Against the Dark Arts" "Divination" "Muggle Studies" "Ancient Runes" "History of Magic" "Transfiguration" "Charms" "Flying"
    MLKit.CommandLine.register_flag("X", description="The features to display.", has_multiple_values=True)
    MLKit.CommandLine.register_flag("Y", description="The target column of the values to display.")
    MLKit.CommandLine.register_usage("logreg_train.py [csv_file_name]\nDisplay a pair plot of a csv data set.")
    MLKit.CommandLine.show_usage_if_needed()

    file_name = MLKit.CommandLine.get_argument_at_index(1)
    target_column_name = MLKit.CommandLine.get_value_for_flag("Y")
    features = MLKit.CommandLine.get_value_for_flag("X")

    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()    
    data_table.display_pair_plot(target_column_name, features)
