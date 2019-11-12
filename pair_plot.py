import MLKit

if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_atributes()

    features = ["Astronomy", "Herbology", "Ancient Runes"]
    
    data_table.display_pair_plot("Hogwarts House", features)
