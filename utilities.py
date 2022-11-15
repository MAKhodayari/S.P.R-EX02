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


def convert(info):
    m_sample = len(info) // 28
    data = []
    for i in range(m_sample):
        str_digit = info[i * 28: (i + 1) * 28]
        int_digit = np.zeros((28, 28), int)
        for j in range(28):
            for k in range(28):
                if str_digit[j][k] != ' ':
                    int_digit[j][k] = 1
        data.append(int_digit)
    return data


def open_naive_bayes():
    with open('./dataset/digitdata/trainingimages', 'r') as training_image:
        train_info = []
        for line in training_image.readlines():
            train_info.append(line[:-1])
    with open('./dataset/digitdata/traininglabels', 'r') as training_label:
        train_label = []
        for line in training_label.readlines():
            train_label.append(list(map(int, line[:-1]))[0])
    with open('./dataset/digitdata/testimages', 'r') as testing_image:
        test_info = []
        for line in testing_image.readlines():
            test_info.append(line[:-1])
    with open('./dataset/digitdata/testlabels', 'r') as testing_label:
        test_label = []
        for line in testing_label.readlines():
            test_label.append(list(map(int, line[:-1]))[0])
    train_data = convert(train_info)
    test_data = convert(test_info)
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)


def class_probability(label):
    m_sample = len(label)
    _, count = np.unique(label, return_counts=True)
    class_prob = count / m_sample
    return class_prob


def index_probability(data, label, s_value):
    unique, count = np.unique(label, return_counts=True)
    m_sample = len(label)
    n_feature = 28 * 28
    c_class = len(unique)
    index_prob = np.zeros((n_feature, c_class, 2))
    for m in range(m_sample):
        for i in range(28):
            for j in range(28):
                if data[m][i][j] == 0:
                    index_prob[28 * i + j][label[m]][0] += 1
                elif data[m][i][j] == 1:
                    index_prob[28 * i + j][label[m]][1] += 1
    for n in range(n_feature):
        for c in range(c_class):
            index_prob[n][c] = (index_prob[n][c] + s_value) / (count[c] + (n_feature * s_value))
    return index_prob


def naive_bayes_prediction(data, c_prob, i_prob):
    c_class = len(c_prob)
    m_sample = len(data)
    data_pred = np.zeros((m_sample, 1), int)
    for i, d in enumerate(data):
        pred_arr = c_prob.copy()
        for j in range(28):
            for k in range(28):
                for m in range(c_class):
                    if d[j][k] == 0:
                        pred_arr[m] *= i_prob[28 * j + k][m][0]
                    elif d[j][k] == 1:
                        pred_arr[m] *= i_prob[28 * j + k][m][1]
        data_pred[i][0] = np.argmax(pred_arr)
    return data_pred
