import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def plotData(X,Y):
    males = [[]]
    females = [[]]

    print('Plotting graphs')
    i = 0
    for i in range(len(Y)):
        if (i%3 == 0):
            if Y[i] == 'male':
                if (len(X[i][2]) < 140):
                    males.append([len(X[i][1]), len(X[i][2])])
            elif Y[i] == 'female':
                if (len(X[i][2]) < 140):
                    females.append([len(X[i][1]), len(X[i][2])])

    fig, ax = plt.subplots()

    males.pop(0)
    females.pop(0)

    x1 = []
    x2 = []

    for male in males:
        x1.append(male[0])
        x2.append((male[1]))

    y1 = []
    y2 = []

    for female in females:
        y1.append(female[0])
        y2.append((female[1]))

    ax.scatter(x1, x2, c='blue', marker='o', label='Male', s = 2)
    ax.scatter(y1, y2, c='pink', marker='o', label='Female', s = 2)
    ax.set_xlabel('DescriptionLength')
    ax.set_ylabel('TweetLength')
    ax.legend(bbox_to_anchor=(1, 1))
    fig.savefig("DescriptionLength_TweetLength.png", bbox_inches="tight", dpi = 500)
    print("plotData complete")



if __name__ == '__main__':

    from collections import defaultdict
    columns = defaultdict(list)  # each value in each column is appended to a list
    first_line = True

    X =[[]]
    with open('description-text.csv',encoding='Latin-1') as f:
        reader = csv.reader(f, delimiter=',', quotechar=' ')  # read rows into a dictionary format
        for row in reader:
            if first_line:
                first_line = False
            else:
                X.append(row)
        X.pop(0)

    X = np.array(X)
    Y = []
    input = [[]]
    for row in X:
        Y.append(row[0])
        input.append((row[1]))
    print(Y)
    print(len(input))
    plotData(X,Y)


    print("Done!")





