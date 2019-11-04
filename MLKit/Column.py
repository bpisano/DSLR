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
        self.attributes = None
    
    def compute_attributes(self):
        """Compute the column attributes"""
        self.attributes = MLKit.ColumnAttributes(self)
    
    def display_attributes(self):
        """Display the column attributes"""
        

