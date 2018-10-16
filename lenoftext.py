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


def plotData(X,Y):
    males = [[]]
    females = [[]]

    print('Plotting graphs')
    i = 0
    for i in range (len(Y)):
        if Y[i] == 'male':
            males.append([X[i][0], X[i][1]])
        elif Y[i] == 'female':
            females.append([X[i][0], X[i][1]])
    print(len(males))
    print(len(females))
    fig, ax = plt.subplots()

    males.pop(0)
    females.pop(0)

    x1 = []
    x2 = []

    for male in males:
        x1.append(male[0])
        x2.append(int(male[1]))

    y1 = []
    y2 = []

    for female in females:
        y1.append(female[0])
        y2.append(int(female[1]))

    ax.scatter(x1, x2, c='blue', marker='o', label='Male')
    ax.scatter(y1, y2, c='pink', marker='x', label='Female')
    ax.set_xlabel('descriptionlength')
    ax.set_ylabel('tweetlength')
    fig.savefig("graph3.png", bbox_inches="tight")
    print("plotData complete")




docPath = "Dataset/gender-classifier-DFE-791531.csv"
lDataPath = "Dateset/labeldatatext.csv"
mask = [1,3,8] # columns chosen from the ender-classifier-DFE-791531.csv to be represented in labeldata.csv
parseDataset(docPath,lDataPath,mask)

dataset = loadDataset([[]], lDataPath)
for i in range(len(dataset)):
    dataset[i][1] = len(dataset[i][1])
    dataset[i][2] = len(dataset[i][2])
parseDataset(docPath,lDataPath,[0,1,2])
cleanData = [[]]
Y = [[]]

for data in dataset:
    if len(data) != 0:
        cleanData.append([data[0], data[1], data[2]])
        Y.append(data[0])
cleanData.pop(0)
plotData(cleanData)

print("Done!")





