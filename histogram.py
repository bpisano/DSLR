import matplotlib.pyplot as plt
import MLKit
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
    top_dict = dict()
    print(data_table.values_for_target_column_named("Hogwarts House", ["Ravenclaw"], "Herbology", scaled=True))
    # for house in houses:
    #     top_dict[house] = dict()
    #     for feature in features:
    #         top_dict[house][feature] = list()
    # try:
    #     file = open(file_name)
    # except FileNotFoundError:
    #     MLKit.Display.error("Incorrect name of file.")
    # reader = list(csv.DictReader(file))
    # for line in reader:
    #     hogwarts_house = line.get('Hogwarts House')
    #     house_dict = top_dict.get(hogwarts_house)
    #     for feature in features:
    #         if line[feature] is '':
    #             continue
    #         house_dict[feature] += [line[feature]]
    # for feature in features:
    #     for house in houses:
    #         plt.hist(top_dict[house][feature], density=True)
    #
    #     plt.show()
    #     exit()
