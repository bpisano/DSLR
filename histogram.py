import matplotlib.pyplot as plt
import MLKit


if __name__ == "__main__":
    file_name = MLKit.CommandLine.get_file_name()
    data_table = MLKit.DataTable(file_name)
    data_table.compute_columns_atributes()
    
    houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff']
    features = ["Arithmancy", "Astronomy", "Herbology", "Defense Against the Dark Arts", "Divination", "Muggle Studies",
                "Ancient Runes", "History of Magic", "Transfiguration", "Potions", "Care of Magical Creatures", "Charms"
                , "Flying"]
    data_table.display_histogram("Hogwarts House", houses, features)
