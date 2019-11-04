import MLKit


class DataTable:
    
    def __init__(self, file_name):
        file_content = MLKit.FileManager.get_content_of_file(file_name)
        columns_dict = MLKit.FileManager.get_csv_data(file_content)

        self.columns = []

        for (key, value) in columns_dict.items():
            column = MLKit.Column(key, value)
            self.columns.append(column)
    
    def compute_columns_data(self):
        for column in self.columns:
            column.compute_data()

