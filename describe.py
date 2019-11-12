import MLKit

if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()
    data_table.display_attributes(from_index=6)

    print(data_table.values_for_target_column_named("Hogwarts House", ["Ravenclaw"], ["Best Hand"], scaled=True))
