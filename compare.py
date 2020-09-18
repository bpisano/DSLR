import csv

if __name__ == "__main__":
    fd1 = open("dataset/train.csv")
    fd2 = open("dataset/predict.csv")
    rd1 = csv.reader(fd1)
    rd2 = csv.reader(fd2)
    our_result = list()
    good_result = 0
    result = 0
    for val in rd1:
        our_result += [val]
    for index, val in enumerate(rd2):
        if our_result[index] == val:
            good_result += 1
        result += 1
    print(str(good_result/result))
