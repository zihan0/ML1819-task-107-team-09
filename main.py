import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def parseDataset(DocPath, lDataPath, mask):
    # re-writing file labeldata.csv
    print('Parsing dataset')
    f = open(lDataPath, "w+")
    f.truncate()

    with open(DocPath) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        col_data = [None] * len(mask)

        for row in csv_reader:

            ## Selecting masked columns
            count = 0
            for pos in mask:
                col_data[count] = row[pos]
                count = count + 1;

            ##Opening labeldata.csv
            with open(lDataPath, 'a') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(col_data)
    print('Parsing complete')

def loadDataset(dataset, path):

    print("Reading from parsed dataset...")
    first_line = True

    with open(path, newline='', encoding='Latin-1') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in reader:
            if first_line:
                first_line = False
            else:
                dataset.append(row)
    dataset.pop(0)
    return dataset;


def plotData(dataset):
    males = [[]]
    females = [[]]
    print('Plotting graphs')
    max1 = int(dataset[0][1])
    min1 = int(dataset[0][1])
    max2 = int(dataset[0][2])
    min2 = int(dataset[0][2])

    for data in dataset:
        try:
            if data[0] == 'male':
                males.append([data[1], data[2]])
            elif data[0] == 'female':
                females.append([data[1], data[2]])

            if int(data[1]) > max1:
                max1 = int(data[1])
            if int(data[1]) < min1:
                min1 = int(data[1])

            if int(data[2]) > max2:
                max2 = int(data[2])
            if int(data[2]) < min2:
                min2 = int(data[2])

        except:
            # do nothing
            pass

    for data in dataset:
        data[1] = (int(data[1]) - min1) / (max1 - min1)
        data[2] = (int(data[2]) - min2) / (max2 - min2)

    males.pop(0)
    females.pop(0)

    fig, ax = plt.subplots()

    x1 = []
    x2 = []
    for x in males:
        x1.append(int(x[0]))
        x2.append(int(x[1]))

    y1 = []
    y2 = []
    for y in females:
        y1.append(int(y[0]))
        y2.append(int(y[1]))

    ax.scatter(x1, x2, c='blue', marker='o', label='Male')
    ax.scatter(y1, y2, c='pink', marker='x', label='Female')

    ax.set_xlabel('Favorites')
    ax.set_ylabel('Tweets')

    fig.savefig("graph1.png", bbox_inches="tight")
    print("plotData complete")


docPath = "Dataset/gender-classifier-DFE-791531.csv"
lDataPath = "Dataset/labeldata.csv"
mask = [0,1,2,4,5,7,9] # columns chosen from the ender-classifier-DFE-791531.csv to be represented in labeldata.csv
parseDataset(docPath,lDataPath,mask)

dataset = loadDataset([[]], lDataPath)

cleanData = [[]]
Y = [[]]

for data in dataset:
    if len(data) != 0:
        cleanData.append([data[1], data[3], data[6]])
        Y.append(data[1])
cleanData.pop(0)
plotData(cleanData)

print("Done!")
