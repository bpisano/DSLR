import MLKit

if __name__ == "__main__":
    MLKit.CommandLine.register_flag("Y", description="The target column where the values should be predicted.", required=True)
    MLKit.CommandLine.register_flag("m", description="The model file name used to predict values.", required=True)
    MLKit.CommandLine.register_flag("s", description="The csv file name with the predicted values.")
    MLKit.CommandLine.register_usage("logreg_predict.py [csv_file_name]\nPredict the value of a csv data file from a trained model.")
    MLKit.CommandLine.show_usage_if_needed()

    file_name = MLKit.CommandLine.get_argument_at_index(1)
    target_column_name = MLKit.CommandLine.get_value_for_flag("Y")
    model_file_name = MLKit.CommandLine.get_value_for_flag("m")
    output_csv = MLKit.CommandLine.get_value_for_flag("s")

    data_table = MLKit.DataTable(file_name)
    data_table.predict(target_column_name, model_file_name)

    if not output_csv is None:
        data_table.save(output_csv)
