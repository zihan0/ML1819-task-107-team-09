import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

def LoadDataset(dataset):
    firstLine = True

    with open('Dataset/gender-classifier-DFE-791531.csv', newline='', encoding='Latin-1') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in reader:
            if firstLine:
                firstLine = False
            else:
                dataset.append(row)
    dataset.pop(0)
    return dataset;



def plotData(dataset):

    males = [[]]
    females = [[]]

    for data in dataset:
        try:
            if data[1] == 'male':
                males.append([data[4], data[9]])
            elif data[1] == 'female':
                females.append([data[4], data[9]])
        except:
            print()
            #do nothing


    males.pop(0)
    females.pop(0)
    males = males[0:500]

    fig, ax = plt.subplots()

    x1 = []
    x2 = []
    for x in males:
        x1.append(x[0])
        x2.append(x[1])

    y1 = []
    y2 = []
    females = females[0:500]
    for y in females:
        y1.append(y[0])
        y2.append(y[1])

    ax.scatter(x1, x2, c='blue', marker='o', label='Male')
    ax.scatter(y1, y2, c='pink', marker='x', label='Female')

    print("1")
    fig.savefig("graph1.png")
    print("2")

dataset = LoadDataset([[]])
plotData(dataset)


print("Done")

