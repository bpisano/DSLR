import MLKit


class Column:

    """
    A DataTable Column.

    Attributes:
        name        The name of the column.
        values      The values of the column.
        attributes  The attributes of the column. Computed by calling the compute_attributes method.
    """

    def __init__(self, name, values):
        self.name = name
        self.values = values
        self.scaled_values = []
        self.attributes = None
    
    def compute_attributes(self):
        """Compute the column attributes"""
        self.attributes = MLKit.ColumnAttributes(self)
        self.__scale_values()

    def __scale_values(self):
        for value in self.values:
            if value == None:
                self.scaled_values.append(None)
                continue
            
            numeric_value = self.attributes.numeric_value_for_value(value)
            scaled_value = (numeric_value - self.attributes.minimum) / (self.attributes.maximum - self.attributes.minimum)
            self.scaled_values.append(scaled_value)

