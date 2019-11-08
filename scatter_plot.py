import matplotlib.pyplot as plt
import MLKit


if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_attributes()
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
                "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures",
                "Charms", "Flying"]
    data = data_table.values_for_target_column_named("Hogwarts House", houses, features, scaled=True)
    fig, axs = plt.subplots(figsize=(15, 10))
    plt.xlabel("x-label")
    plt.ylabel("y-label")
    plt.axis([0, 1, 0, 1])
    for house in houses:
        while None in data[house][features[7]]:
            del data[house][features[8]][data[house][features[7]].index(None)]
            del data[house][features[7]][data[house][features[7]].index(None)]
        while None in data[house][features[8]]:
            del data[house][features[7]][data[house][features[8]].index(None)]
            del data[house][features[8]][data[house][features[8]].index(None)]
        for index, i in enumerate(data[house][features[7]]):
            data[house][features[7]][index] = float(i)
        for index, i in enumerate(data[house][features[8]]):
            data[house][features[8]][index] = float(i)
        l1 = axs.scatter(data[house][features[7]], data[house][features[8]], label=house)
    fig.legend(houses, loc='upper right')
    plt.show()
