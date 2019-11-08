import MLKit

import time
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
    
    def all_columns(self):
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

    def values_for_target_column_named(self, column_name, value_names, target_column_names, scaled=False):
        """Return a dictionnary of all the values in a target column, corresponding to the row values of a given column."""
        column = self.column_named(column_name)
        value_names = list(map(str, value_names))
        target_column_names = list(map(str, target_column_names))
        values = {}

        for index, value in enumerate(column.values):
            value_str = str(value)

            if value_str in value_names:
                if values.get(value_str) == None:
                    values[value_str] = {}
                
                for target_column_name in target_column_names:
                    if values[value_str].get(target_column_name) == None:
                        values[value_str][target_column_name] = []
                    
                    target_column = self.column_named(target_column_name)
                    fetched_value = target_column.scaled_values[index] if scaled is True else target_column.values[index]
                    int_value = int(fetched_value) if fetched_value != None else None
                    values[value_str][target_column_name].append(int_value)
        
        return values
    
    def compute_columns_atributes(self):
        """Compute the attributes of each column."""
        for column in self.__columns.values():
            column.compute_attributes()
    
    def display_attributes(self, from_index=0, to_index=-1):
        """Display the calculated attributes."""
        columns = self.all_columns()

        if to_index < 0:
            to_index = len(columns)
        if from_index > to_index or from_index > len(columns):
            from_index = 0

        first_column_size = 10
        column_size = 20
        attributes_name = MLKit.ColumnAttributes.all()

        # Column names
        line_str = MLKit.Display.sized_str("", first_column_size)
        for (index, column) in enumerate(columns):
            if not (index >= from_index and index < to_index):
                    continue
            sized_str = MLKit.Display.sized_str(column.name + " ", column_size)
            line_str += MLKit.Display.attributed_str(sized_str, [MLKit.Color.blue])

        print("")
        print(line_str)

        # Line
        width = first_column_size + column_size * (to_index - from_index)
        table_line = MLKit.Display.line_str(width - first_column_size - 1) + " "
        sized_table_line = MLKit.Display.sized_str(table_line, width)
        print(sized_table_line)

        # Attributes
        for attribute_name in attributes_name:
            sized_str = MLKit.Display.sized_str(attribute_name + "|", first_column_size)
            line_str = MLKit.Display.attributed_str(sized_str, [MLKit.Style.bold])
            
            for (index, column) in enumerate(columns):
                if not (index >= from_index and index < to_index):
                    continue

                attribute_value = column.attributes.value_for_key(attribute_name)
                if isinstance(attribute_value, float):
                    if abs(attribute_value) == float("inf"):
                        line_str += MLKit.Display.sized_str("-", column_size)
                    else:
                        line_str += MLKit.Display.sized_str("%.6f|" % attribute_value, column_size)
                elif isinstance(attribute_value, int):
                    line_str += MLKit.Display.sized_str(str(attribute_value) + "|", column_size)
                elif isinstance(attribute_value, str):
                    line_str += MLKit.Display.sized_str(attribute_value + "|", column_size)
                else:
                    line_str += MLKit.Display.sized_str("-|", column_size)
            
            print(line_str)
        
        # Line
        print(sized_table_line)
        print("")
