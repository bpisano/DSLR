import MLKit
import csv


def get_content_of_file(file_name):
    try:
        file_descriptor = open(file_name, "r")
        content = file_descriptor.readlines()
        file_descriptor.close()
        return content
    except IOError:
        MLKit.Display.error("No such file named " + file_name)


def get_csv_data(file_content, delimiter=","):
    if file_content == None:
        return None
    
    columns = dict()
    csv_content = csv.DictReader(file_content, delimiter=delimiter)
    
    for row in csv_content:
        for (key, value) in row.items():
            if columns.get(key) == None:
                columns[key] = []
            try:
                columns[key].append(value)
            except TypeError:
                MLKit.Display.error("The file is not correctly formated.")

    return columns
