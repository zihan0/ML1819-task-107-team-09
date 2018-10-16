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
    ax.set_xlabel('color')
    ax.set_ylabel('count')
    fig.savefig("graph2.png", bbox_inches="tight")
    print("plotData complete")


if __name__ == '__main__':

    from collections import defaultdict
    columns = defaultdict(list)  # each value in each column is appended to a list
    with open('n_3.csv') as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            for (k, v) in row.items():  # go over each column name and value
                columns[k].append(v)
    X = [columns['color'], columns['count'],columns['gender']]
    X = np.asarray(X).T  # change list to array X.shape=(12894, 2)
    #X = colortoflag(X)
    #print(X)
    Y = X[:,2]
    #print(Y)
    plotData(X,Y)


