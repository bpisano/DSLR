import MLKit


class DataTable:

    """
    A representation of a csv file.
    """
    
    def __init__(self, file_name):
        file_content = MLKit.FileManager.get_content_of_file(file_name)
        columns_dict = MLKit.FileManager.get_csv_data(file_content)

        self.__columns = {}

        for (name, values) in columns_dict.items():
            column = MLKit.Column(name, values)
            self.__columns[name] = column
    
    def get_columns(self):
        """Return the columns of the DataTable."""
        return list(self.__columns.values())
    
    def column_named(self, column_name):
        """Return the column for a given column name."""
        return self.__columns.get(column_name)

    def values_for_column_named(self, column_name):
        """Return the values of a column for a given column name."""

        column = self.column_named(column_name)

        if column == None:
            MLKit.Display.warning("No such column named " + column_name + " in this data table.")
            return []
        else:
            return column.values
    
    def compute_columns_atributes(self):
        """Compute the attributes of each column."""
        for column in self.__columns.values():
            column.compute_attributes()
    
    def display_attributes(self, from_index=0, to_index=-1):
        """Display the calculated attributes."""
        columns = self.get_columns()

        if to_index < 0:
            to_index = len(columns)
        if (from_index > to_index):
            from_index = 0

        first_column_size = 10
        column_size = 20
        attributes_name = MLKit.ColumnAttributes.all()

        line_str = MLKit.Display.sized_str("", first_column_size)
        for (index, column) in enumerate(columns):
            if not (index >= from_index and index < to_index):
                    continue
            line_str += MLKit.Display.sized_str(column.name, column_size)

        print(line_str)

        for attribute_name in attributes_name:
            line_str = MLKit.Display.sized_str(attribute_name, first_column_size)
            
            for (index, column) in enumerate(columns):
                if not (index >= from_index and index < to_index):
                    continue

                attribute_value = column.attributes.value_for_key(attribute_name)
                if isinstance(attribute_value, float):
                    if abs(attribute_value) == float("inf"):
                        line_str += MLKit.Display.sized_str("No value", column_size)
                    else:
                        line_str += MLKit.Display.sized_str("%.6f" % attribute_value, column_size)
                elif isinstance(attribute_value, int):
                    line_str += MLKit.Display.sized_str(str(attribute_value), column_size)
                elif isinstance(attribute_value, str):
                    line_str += MLKit.Display.sized_str(attribute_value, column_size)
                else:
                    line_str += MLKit.Display.sized_str("No value", column_size)
            
            print(line_str)


