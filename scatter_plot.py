import matplotlib.pyplot as plt
import MLKit
import copy
import numpy
import sys
import csv

if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_atributes()
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
                "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms"
                , "Flying"]
    data = data_table.values_for_target_column_named("Hogwarts House", houses, features)
    fig, axs = plt.subplots(figsize=(15, 10))
    for house in houses:
        remove = list()
        while None in data[house][features[0]]:
            del data[house][features[1]][data[house][features[0]].index(None)]
            del data[house][features[0]][data[house][features[0]].index(None)]
        while None in data[house][features[1]]:
            del data[house][features[0]][data[house][features[1]].index(None)]
            del data[house][features[1]][data[house][features[1]].index(None)]
        for index, i in enumerate(data[house][features[0]]):
            data[house][features[0]][index] = float(i)
        for index, i in enumerate(data[house][features[1]]):
            data[house][features[1]][index] = float(i)
        print(data[house][features[0]])
                # print(data[house][features[1]])
        l1 = axs.scatter(data[house][features[0]], data[house][features[1]], label=house)
    fig.legend(houses, loc='upper right')
    plt.show()
