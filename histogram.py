import matplotlib.pyplot as plt
import MLKit


if __name__ == "__main__":

    data_table = MLKit.DataTable(MLKit.CommandLine.get_file_name())
    data_table.compute_columns_attributes()
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
                "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures",
                "Charms", "Flying"]
    data = data_table.values_for_target_column_named("Hogwarts House", houses, features, scaled=True)
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(15, 10))
    for index, feature in enumerate(features):
        for house in houses:
            data[house][feature] = [x for x in data[house][feature] if x is not None]
            axs[int(index/4)][index % 4].hist(data[house][feature], alpha=0.4, label=house)
        axs[int(index/4)][index % 4].title.set_text(feature)
    fig.delaxes(axs[3][1])
    fig.delaxes(axs[3][2])
    fig.delaxes(axs[3][3])
    fig.tight_layout()
    fig.legend(houses, loc='lower right', ncol=5)
    plt.show()
