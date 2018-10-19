import numpy as np
import matplotlib.pyplot as plt
import csv

def plotData(X, Y):
    males = [[]]
    females = [[]]
    print('Plotting graphs')
    i = 0
    for i in range(len(Y)):
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



def predict(X, theta):
    # calculates the prediction h_theta(x) for input(s) x contained in array X
    ##### replace the next line with your code #####
    pred = np.sign(np.dot(X, theta))
    return pred


def computeCost(X, y, theta):
    # function calculates the cost J(theta) and returns its value
    ##### replace the next line with your code #####
    length = len(y)
    cost = 0
    for i in range(length):
        cost = cost + np.log(1 + np.exp(-(np.dot(np.dot(y[i], theta.T), X[i]))))
    return cost/length


def computeGradient(X, y, theta):
    # calculate the gradient of J(theta) and return its value
    ##### replace the next lines with your code #####
    n = len(theta)
    m = len(X)
    grad = np.zeros(n)
    for i in range(m):
        a = np.exp(-(np.dot(np.dot(y[i], theta.T), X[i])))
        grad = grad - ((np.dot(y[i], X[i])) * (a / (1 + a)))
    return grad / m


def gradientDescent(X, y):
    # iteratively update parameter vector theta
    # initialize variables for learning rate and iterations
    alpha = 0.1
    iters = 10000
    cost = np.zeros(iters)
    (m, n) = X.shape
    theta = np.zeros(n)

    for i in range(iters):
        theta = theta - alpha * computeGradient(X, y, theta)
        cost[i] = computeCost(X, y, theta)

    return theta, cost


def normaliseData(x):
    # rescale data to lie between 0 and 1
    scale = x.max(axis=0)
    return (x / scale, scale)


def addQuadraticFeature(X):
    # Given feature vector [x_1,x_2] as input, extend this to
    # [x_1,x_2,x_1*x_1] i.e. add a new quadratic feature
    ##### insert your code here #####
    (m, n) = X.shape
    Xt = np.zeros((m, n + 1))
    for i in range(m):
        Xt[i][2] = X[i][0] * X[i][0]
        for j in range(n):
            Xt[i][j] = X[i][j]
    return Xt


def computeScore(X, y, preds):
    # for training data X,y it calculates the number of correct predictions made by the model
    ##### replace the next line with your code #####
    score = 0
    length = len(y)
    for i in range(length):
        if preds[i] == y[i]:
            score = score + 1
        else:
            continue
    return score


def plotDecisionBoundary(Xt, y, Xscale, theta):
    # plots the training data plus the decision boundary in the model
    fig, ax = plt.subplots(figsize=(12, 8))
    # plot the data
    positive = y > 0
    negative = y < 0
    ax.scatter(Xt[positive, 1] * Xscale[1], Xt[positive, 2] * Xscale[2], c='b', marker='o', label='Healthy')
    ax.scatter(Xt[negative, 1] * Xscale[1], Xt[negative, 2] * Xscale[2], c='r', marker='x', label='Not Healthy')
    # calc the decision boundary
    x = np.linspace(Xt[:, 2].min() * Xscale[2], Xt[:, 2].max() * Xscale[2], 50)
    if (len(theta) == 3):
        # linear boundary
        x2 = -(theta[0] / Xscale[0] + theta[1] * x / Xscale[1]) / theta[2] * Xscale[2]
    else:
        # quadratic boundary
        x2 = -(theta[0] / Xscale[0] + theta[1] * x / Xscale[1] + theta[3] * np.square(x) / Xscale[3]) / theta[2] * \
             Xscale[2]
    # and plot it
    ax.plot(x, x2, label='Decision boundary')
    ax.legend()
    ax.set_xlabel('Test 1')
    ax.set_ylabel('Test 2')
    fig.savefig('pred.png')


def main():
    # load the training data
    from collections import defaultdict
    X = [[]]
    columns = defaultdict(list)  # each value in each column is appended to a list
    with open('fav_count.csv') as f:
        reader = csv.DictReader(f)  # read rows into a dictionary format
        for row in reader:  # read a row as {column1: value1, column2: value2,...}
            for (k, v) in row.items():  # go over each column name and value
                (columns[k].append(v))
    X = [columns['0'], columns['12748'], columns['male']]
    X = np.asarray(X).T  # change list to array X.shape=(12894, 2)



    for i in range(len(X)):
        if (X[i][2] == 'male'): X[i][2] = 1
        else:X[i][2] = 0
    X = np.array(X).astype(np.float)
    y = X[:, 2]
    print(y)

#    X = addQuadraticFeature(X)

    # plot it so we can see how it looks
    #plotData(X, y)

    # add a column of ones to input data
    m = len(y)
    Xt = np.column_stack((np.ones((m, 1)), X))
    (m, n) = Xt.shape  # m is number of data points, n number of features
    Xt = np.array(Xt).astype(np.float)
    # rescale training data to lie between 0 and 1
    (Xt, Xscale) = normaliseData(Xt)

    # calculate the cost when theta is zero

    print('testing the cost function ...')
    theta = np.zeros(n)
    print('when theta is zero cost = ', computeCost(Xt, y, theta))
    input('Press Enter to continue...')

    # calculate the gradient when theta is zero
    print('testing the gradient function ...')
    print('when theta is zero gradient = ', computeGradient(Xt, y, theta))
    input('Press Enter to continue...')

    # perform gradient descent to "fit" the model parameters
    print('running gradient descent ...')
    theta, cost = gradientDescent(Xt, y)
    print('after running gradientDescent() theta=', theta)

    # plot the prediction
    plotDecisionBoundary(Xt, y, Xscale, theta)

    preds = predict(Xt, theta)
    score = computeScore(Xt, y, preds)
    print('accuracy = {0:.2f}%'.format(score / len(y) * 100))
    
    # plot how the cost varies as the gradient descent proceeds
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2.semilogy(cost, 'r')
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('cost')
    fig2.savefig('cost.png')


if __name__ == '__main__':
    main()
