import MLKit


class Column:

    """
    A DataTable Column.

    Attributes:
        name        The name of the column.
        values      The values of the column.
    """

    def __init__(self, name, values):
        self.name = name
        self.values = values
    
    def compute_data(self):
        length = 0
        values_sum = 0
        minimum = -float("inf")
        maximum = float("inf")

        # for value in self.values:
            
