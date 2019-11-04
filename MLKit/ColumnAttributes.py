import math
import MLKit


class ColumnAttributes:

    def __init__(self, column):
        self.count = 0
        self.mean = None
        self.minimum = float("inf")
        self.maximum = -float("inf")
        self.percent_25 = None
        self.percent_50 = None
        self.percent_75 = None
        self.std = None
        values_sum = 0

        for value in column.values:
            if len(value) == 0:
                continue

            try:
                float(value)
            except ValueError:
                # To be improved.
                # Should replace every string values by a numeric one.
                MLKit.Display.warning("Cannot compute attributes in column " + column.name + " because the values are not numeric.")
                return

            float_value = float(value)
            
            self.count += 1
            values_sum += float_value

            if float_value < self.minimum:
                self.minimum = float_value
            if float_value > self.maximum:
                self.maximum = float_value
        
        if self.count == 0:
            MLKit.Display.warning("Column " + column.name + " doesn't have any values.")
            return

        # median = self.__mediane_from_values(column.values, length)
        self.mean = values_sum / self.count
        self.percent_25 = 0
        self.percent_50 = 0
        self.percent_75 = 0
        squared_sum = 0

        for (index, value) in enumerate(column.values):
            if len(value) == 0:
                continue

            float_value = float(value)
            squared_sum += (self.mean - float_value) ** 2
            
            if index <= self.count / 4:
                self.percent_25 = float_value
            if index <= self.count / 2:
                self.percent_50 = float_value
            if index <= 3 * self.count / 4:
                self.percent_75 = float_value

        variance = squared_sum / self.count
        self.std = math.sqrt(variance)
    
    # def __mediane_from_values(self, values, length):
    #     if length % 2 == 0:
    #         value_1 = values[length / 2]
    #         value_2 = values[length / 2 + 1]
    #         return (value_1 + value_2) / 2
    #     else:
    #         return values[(length - 1) / 2]

    def value_for_key(self, name):
        if name == "Count":
            return self.count
        elif name == "Mean":
            return self.mean
        elif name == "Std":
            return self.std
        elif name == "Min":
            return self.minimum
        elif name == "25%":
            return self.percent_25
        elif name == "50%":
            return self.percent_50
        elif name == "75%":
            return self.percent_75
        elif name == "Max":
            return self.maximum
        else:
            return None

    @staticmethod
    def all():
        return ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"]
        
