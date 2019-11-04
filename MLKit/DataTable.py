import MLKit


class DataTable:
    
    def __init__(self, file_name):
        file_content = MLKit.FileManager.get_content_of_file(file_name)
        columns_dict = MLKit.FileManager.get_csv_data(file_content)

        self.__columns = {}

        for (key, value) in columns_dict.items():
            column = MLKit.Column(key, value)
            self.__columns[key] = column
    
    def column_named(self, column_name):
        """Return the column for a given column name."""
        return self.__columns.get(column_name)

    def values_for_column_named(self, column_name):
        """Return the values for a column for a given column name."""

        column = self.column_named(column_name)

        if column == None:
            MLKit.Display.warning("No such column named " + column_name + " in this data table.")
            return []
        else:
            return column.values
    
    # def compute_columns_data(self):
    #     for column in self.columns:
    #         column.compute_data()

