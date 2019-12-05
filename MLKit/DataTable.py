import MLKit
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import operator
from sklearn.metrics import accuracy_score


class DataTable:
    """
    A representation of a csv file.
    """

    def __init__(self, file_name):
        self.file_name = file_name
        file_content = MLKit.FileManager.get_content_of_file(file_name)
        columns_dict = MLKit.FileManager.get_csv_data(file_content)

        self.__columns = {}
        self.train_conditions = None

        for (name, values) in columns_dict.items():
            column = MLKit.Column(name, values)
            self.__columns[name] = column

    def __str__(self):
        self.display_attributes()
        return ""

    def all_columns(self):
        """Return the columns of the DataTable."""
        return list(self.__columns.values())

    def column_named(self, column_name):
        """Return the column for a given column name."""
        return self.__columns.get(column_name)

    def values_for_column_named(self, column_name):
        """Return the values of a column for a given column name."""

        column = self.column_named(column_name)

        if column is None:
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

        if column is None:
            MLKit.Display.error("Column " + column_name + " doesn't exists.")

        for index, value in enumerate(column.values):
            value_str = str(value)

            if value_str in value_names:
                if values.get(value_str) is None:
                    values[value_str] = {}

                for target_column_name in target_column_names:
                    if values[value_str].get(target_column_name) is None:
                        values[value_str][target_column_name] = []

                    target_column = self.column_named(target_column_name)
                    if target_column is None:
                        MLKit.Display.error("Column " + target_column_name + " doesn't exists.")

                    if scaled is True:
                        try:
                            float_value = float(target_column.scaled_values[index])
                            values[value_str][target_column_name].append(float_value)
                        except TypeError:
                            values[value_str][target_column_name].append(None)
                    else:
                        try:
                            float_value = float(target_column.values[index])
                            values[value_str][target_column_name].append(float_value)
                        except TypeError:
                            values[value_str][target_column_name].append(target_column.values[index])

        return values

    def feature_values_for_rows_in_target_column(self, target_column_name, row_names, feature_column_names):
        data = {}
        target_column = self.column_named(target_column_name)

        for row_name in row_names:
            data[row_name] = []
            for target_column_index, target_column_value in enumerate(target_column.values):
                if not target_column_value == row_name:
                    continue

                row_contains_none = False

                for feature_column_name in feature_column_names:
                    feature_column = self.column_named(feature_column_name)
                    if feature_column.values[target_column_index] == None:
                        row_contains_none = True
                        break

                if row_contains_none == True:
                    continue

                row_columns = {}

                for feature_column_name in feature_column_names:
                    feature_column = self.column_named(feature_column_name)
                    feature_column_value = feature_column.values[target_column_index]

                    try:
                        float_value = float(feature_column_value)
                        row_columns[feature_column_name] = float_value
                    except TypeError:
                        MLKit.Display.error("Value for column " + feature_column_name + " should be numeric.")

                data[row_name].append(row_columns)

        return data

    def compute_columns_attributes(self):
        """Compute the attributes of each column."""
        for column in self.__columns.values():
            column.compute_attributes()

    def set_train_condition(self, target_column_name, features_column_names):
        """Define the feature columns that will be used for train based on the row values of the target column."""
        target_column = self.column_named(target_column_name)
        row_names = []

        for row_value in target_column.values:
            if not row_value in row_names:
                row_names.append(row_value)

        for row_name in row_names:
            self.add_train_condition(row_name, features_column_names)

    def add_train_condition(self, row_name, features_column_names):
        """Define the features column that will be used for train based on a single row value of the target column."""
        if self.train_conditions is None:
            self.train_conditions = {}

        self.train_conditions[row_name] = features_column_names

    def train(self, target_column_name, features_column_names, file_name, learning_rate=0.1, accuracy_split=None):
        if not 1 >= learning_rate > 0:
            MLKit.Display.error("Learning rate should be greater than 0 and smaller than 1.")

        if accuracy_split is not None and (accuracy_split <= 0 or accuracy_split >= 1):
            MLKit.Display.error("Accuracy split should be greater than 0 and smaller than 1.")

        if self.column_named(target_column_name) is None:
            MLKit.Display.error("Column " + target_column_name + " doesn't exists.")

        for feature_column_name in features_column_names:
            if self.column_named(feature_column_name) is None:
                MLKit.Display.error("Column " + feature_column_name + " doesn't exists.")

        target_column = self.column_named(target_column_name)
        feature_columns = [self.column_named(column_name) for column_name in list(self.__columns.keys()) if
                           column_name in features_column_names]

        X = [column.values for column in feature_columns]
        for index, _ in enumerate(X):
            column = self.column_named(features_column_names[index])
            # print(column.attributes.mean)
            X[index] = [(value or column.attributes.mean) for value in X[index]]
        X = np.asarray(X).astype('float')
        Y = np.asarray(target_column.values)
        feature_names = [column.name for column in feature_columns]

        if accuracy_split is None:
            regression = MLKit.LogisticRegression(learning_rate)
            regression.fit(X, Y, feature_names)
            regression.save(file_name)
        else:
            splited_X = X[:, :int(X.shape[1] * accuracy_split)]
            splited_Y = Y[:int(Y.shape[0] * accuracy_split)]
            splited_test_X = X[:, int(X.shape[1] * accuracy_split):]
            splited_test_Y = Y[int(Y.shape[0] * accuracy_split):]
            regression = MLKit.LogisticRegression(learning_rate)
            regression.fit(splited_X, splited_Y, feature_names)
            regression.save(file_name)

            model = MLKit.FileManager.get_model_data(file_name + ".mlmodel")
            predicted_values = []
            for row_index in range(splited_test_Y.shape[0]):
                predicted_value = self.__predcited_value(splited_test_X, row_index, model)
                predicted_values.append(predicted_value)

            print("Accuracy:", accuracy_score(splited_test_Y, predicted_values))

        MLKit.Display.success("model saved as " + file_name + ".mlmodel")

    def predict(self, target_column_name, model_file_name):
        """Predict values of a target column from a .mlmodel file."""
        model = MLKit.FileManager.get_model_data(model_file_name)
        feature_column_names = list(model[list(model.keys())[0]].keys())[1:]
        target_column = self.column_named(target_column_name)
        feature_columns = [self.column_named(column_name) for column_name in feature_column_names]

        if self.column_named(target_column_name) is None:
            MLKit.Display.error("Column " + target_column_name + " doesn't exists.")

        for feature_column_name in feature_column_names:
            if self.column_named(feature_column_name) is None:
                MLKit.Display.error("Column " + feature_column_name + " doesn't exists.")

        X = np.array([column.values for column in feature_columns])
        for row_index in range(X.shape[1]):
            target_column.values[row_index] = self.__predcited_value(X, row_index, model)

        MLKit.Display.success("Predicted values")

    def __predcited_value(self, X, row_index, model):
        row_probabilities = {}

        for row_name in model.keys():
            row_probabilities[row_name] = 0
            for column_index, column_name in enumerate(model[row_name].keys()):
                if column_name == "t0":
                    row_probabilities[row_name] += model[row_name]["t0"]
                    continue
                else:
                    column_theta = model[row_name][column_name]
                    column_value = X[column_index - 1][row_index]
                    float_column_value = 0 if column_value is None else float(column_value)
                    row_probabilities[row_name] += column_theta * float_column_value

            row_probabilities[row_name] = MLKit.LogisticRegression.predict(row_probabilities[row_name])

        sorted_row_probabilities = sorted(row_probabilities.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_row_probabilities[0][0]

    def save(self, file_name=None):
        """Update the current csv file or create a new one if a file name is provided."""
        final_string = ""

        for row_index in range(len(list(self.__columns.items())[0][1].values)):
            for column_name in self.__columns.keys():
                column = self.column_named(column_name)
                value = column.values[row_index]

                if value is None:
                    final_string += ","
                else:
                    final_string += str(value) + ","

            final_string = final_string[:-1]
            final_string += "\n"

        if file_name is None:
            MLKit.FileManager.save_string(final_string, self.file_name)
            MLKit.Display.success("Saved csv file in " + self.file_name)
        else:
            if ".csv" in file_name:
                MLKit.FileManager.save_string(final_string, file_name)
                MLKit.Display.success("Saved csv file in " + file_name)
            else:
                MLKit.FileManager.save_string(final_string, file_name + ".csv")
                MLKit.Display.success("Saved csv file in " + file_name + ".csv")

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
            if not (from_index <= index < to_index):
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
                if not (from_index <= index < to_index):
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

    def display_histogram(self, target_column, row_names, feature_names, scaled=True):
        """Display an histogram for rows in columns."""
        column_len = len(feature_names)
        n_columns = 4 if column_len > 4 else column_len
        n_rows = column_len / 4

        if column_len % 4 != 0:
            n_rows += 1

        data = self.values_for_target_column_named(target_column, row_names, feature_names, scaled=scaled)
        fig, axs = plt.subplots(nrows=int(n_rows), ncols=int(n_columns), figsize=(15, 10))

        for index, column_name in enumerate(feature_names):
            for row in row_names:
                try:
                    data[row][column_name] = [x for x in data[row][column_name] if x is not None]
                except KeyError:
                    MLKit.Display.error("Value " + row + " doesn't exist in column " + target_column)

                if int(n_rows) == 1:
                    if column_len == 1:
                        axs.hist(data[row][column_name], alpha=0.4, label=row)
                    else:
                        axs[index].hist(data[row][column_name], alpha=0.4, label=row)
                else:
                    axs[int(index / 4)][index % 4].hist(data[row][column_name], alpha=0.4, label=row)
            if int(n_rows) == 1:
                if column_len == 1:
                    axs.title.set_text(column_name)
                else:
                    axs[index].title.set_text(column_name)
            else:
                axs[int(index / 4)][index % 4].title.set_text(column_name)

        if column_len > 4 and column_len % 4 != 0:
            for i in range(column_len % 4, 4):
                fig.delaxes(axs[int(n_rows) - 1][i])

        fig.tight_layout()
        fig.legend(row_names, loc="lower right", ncol=5)
        plt.show()

    def display_pair_plot(self, target_column_name, feature_names):
        csv_data = pd.read_csv(self.file_name)
        csv_data.dropna(axis=0, how="any", inplace=True)

        if self.column_named(target_column_name) is None:
            MLKit.Display.error("Column " + target_column_name + " doesn't exists.")

        for feature_name in feature_names:
            if self.column_named(feature_name) is None:
                MLKit.Display.error("Column " + feature_name + " doesn't exists.")

        sns.pairplot(csv_data, vars=feature_names, hue=target_column_name, diag_kind="hist", height=3)
        plt.show()
