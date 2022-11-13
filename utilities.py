import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize(X):
    NX = pd.DataFrame(columns=X.columns.values)
    for column in NX.columns:
        X_max = X[column].max()
        X_min = X[column].min()
        X_range = X_max - X_min
        if X_range != 0:
            NX[column] = (X[column] - X_min) / X_range
        else:
            NX[column] = X[column] / X_max
    return NX


def open_logistic():
    logistic_data = pd.read_csv('./dataset/seed.txt', sep='\t', names=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'y'])
    return logistic_data


def sigmoid(z):
    h = 1 / (1 + np.exp(-z))
    return h


def logistic_prediction(X, theta, threshold=0.5, norm=False):
    z = np.dot(X, theta.T)
    h = sigmoid(z)
    if not norm:
        return h
    else:
        if threshold == 0.5:
            yh = np.array([1 if label >= threshold else 0 for label in h]).reshape(-1, 1)
        else:
            yh = np.array([1 if label >= threshold else 0 for label in h]).reshape(-1, 1)
        return yh


def calc_cross_entropy(X, y, theta):
    m_sample = X.shape[0]
    ones = np.ones(m_sample)
    h = logistic_prediction(X, theta)
    ce = -(np.dot(y.T, np.log(h)) + np.dot((ones - y).T, np.log(ones - h))) / m_sample
    return ce[0]


def logistic_gradient(X, y, alpha, n_iter):
    m_sample, n_feature = X.shape
    theta = np.random.rand(n_feature).reshape(-1, n_feature)
    iter_list = [i for i in range(n_iter)]
    iter_cost = []
    for i in range(n_iter):
        predictions = logistic_prediction(X, theta)
        change = []
        for j in range(n_feature):
            change.append((np.dot(X.iloc[:, j], (predictions - y))) / m_sample)
            theta[0][j] = theta[0][j] - alpha * change[j]
        cost = sum(change)
        iter_cost.append(cost)
    # print(f'iter {i} | cost {iter_cost[i]}')
    iter_cost = iter_cost[50:]
    iter_list = iter_list[50:]
    plt.plot(iter_list, iter_cost, color='red')
    plt.scatter(iter_list[::250], iter_cost[::250], marker='.', color='black')
    plt.suptitle('Gradient Method')
    plt.title('Cost Per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.show()
    return theta[0]


def calc_accuracy(y, yh):
    m_sample = len(y)
    correct = 0
    for i in range(m_sample):
        if yh[i] == y[i]:
            correct += 1
    acc = correct / m_sample
    return acc
