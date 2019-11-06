import matplotlib
import MLKit
import csv


if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    try :
        f = open(file_name)
    except FileNotFoundError:
        MLKit.Display.error("Incorrect name of file.")
    reader = list(csv.DictReader(f))
    for line in reader:
        hogwarts_house = line.get('Hogwarts House')

