import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def colortoflag(X):
    flag = 100
    color = X[:,0]
    uniqcolor = np.unique(color, return_inverse=True) #376 unique color
    #print(len(uniqcolor[0]))
    for i in range (len(color)):
        temp = np.where(X == color[i])
        #print (temp)
        for j in range (len(temp[0])):
            X[temp[0][j]] = flag
            #X = np.delete(X, (tmp[j]),axis = 1)
        flag = flag * 2
    print(X)
    return X

def hex2rgb(value):
    """Return (red, green, blue) for the color given as #rrggbb."""
    try:
        value = value.lstrip('#')
        lv = len(value)
        tup = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        return tup[2]
    except:
        return 0


def plotData(X,Y):
    males = [[]]
    females = [[]]

    print('Plotting graphs')
    i = 0
    for i in range (len(Y)):
        if Y[i] == 'male':
            males.append([int(X[i][0]), int(X[i][1])])
        elif Y[i] == 'female':
            females.append([int(X[i][0]), int(X[i][1])])
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

    ax.scatter(x1, x2, c='blue', marker='o', label='Male', s=2)
    ax.scatter(y1, y2, c='pink', marker='o', label='Female', s=2)
    ax.set_xlabel('background colour')
    ax.set_ylabel('link colour')
    ax.legend()
    fig.savefig("graph2.png", bbox_inches="tight", dpi=500)
    print("plotData complete")


if __name__ == '__main__':

    from collections import defaultdict
    columns = defaultdict(list)  # each value in each column is appended to a list
    with open('n_3.csv') as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            for (k, v) in row.items():  # go over each column name and value
                columns[k].append(v)
    X = [columns['color'], columns['link_color'], columns['gender']]
    X = np.asarray(X).T  # change list to array X.shape=(12894, 2)

    for x in X:
        x[0] = int(hex2rgb(x[0]))
        x[1] = int(hex2rgb(x[1]))

    Y = X[:,2]
    #print(Y)
    plotData(X,Y)


