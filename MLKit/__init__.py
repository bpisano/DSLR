from MLKit import FileManager


class Column:

    def __init__(self, name, values):
        self.name = name
        self.values = values
    
    def compute_data(self):
        for value in self.values:
            pass



class DataTable:
    
    def __init__(self, file_name):
        file_content = FileManager.get_content_of_file(file_name)
        columns_dict = FileManager.get_csv_data(file_content)

        self.columns = []

        for (key, value) in columns_dict.items():
            column = Column(key, value)
            self.columns.append(column)
    
    def compute_columns_data(self):
        for column in self.columns:
            column.compute_data()

