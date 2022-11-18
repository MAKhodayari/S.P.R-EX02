import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Global Helpers
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


def calc_accuracy(y, yh):
    m_sample = len(y)
    correct = 0
    for i in range(m_sample):
        if yh[i] == y[i]:
            correct += 1
    acc = correct / m_sample
    return acc


# Multiclass (One VS One, One VS All) Helpers
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


def logistic_gradient(X, y, alpha, n_iter):
    m_sample, n_feature = X.shape
    theta = np.random.rand(n_feature).reshape(-1, n_feature)
    iter_cost = []
    for i in range(n_iter):
        pred = logistic_prediction(X, theta)
        change = []
        for j in range(n_feature):
            change.append((np.dot(X.iloc[:, j], (pred - y))) / m_sample)
            theta[0][j] = theta[0][j] - alpha * change[j]
        cost = abs(sum(change))
        iter_cost.append(cost)
    iter_list = [num for num in range(len(iter_cost))]
    plt.figure(figsize=(8, 6))
    plt.plot(iter_list, iter_cost, color='red')
    plt.suptitle('Gradient Method')
    plt.title('Cost Per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.figtext(0.5, 0.01, f'Finished With Cost {iter_cost[-1][0].round(5)} In {i + 1} Iterations', ha='center')
    plt.show()
    return theta[0]


def calc_cross_entropy(X, y, theta):
    m_sample = X.shape[0]
    ones = np.ones(m_sample)
    h = logistic_prediction(X, theta)
    ce = -(np.dot(y.T, np.log(h)) + np.dot((ones - y).T, np.log(ones - h))) / m_sample
    return ce[0]


# Softmax Helpers
def one_hot_encode(y):
    m_sample = len(y)
    classes = np.unique(y)
    c_class = len(classes)
    one_hot = np.zeros((m_sample, c_class), int)
    for i in range(m_sample):
        for j, c in enumerate(classes):
            if y[i] == c:
                one_hot[i][j] = 1
    return one_hot


def softmax_prediction(X, theta, norm=False):
    z = np.dot(X, theta)
    h = np.exp(z - np.max(z, axis=1, keepdims=True))
    yh = h / np.sum(h, axis=1, keepdims=True)
    if not norm:
        return yh
    else:
        return np.argmax(yh, axis=1, keepdims=True) + 1


def softmax_gradient(X, y, alpha, n_iter):
    m_sample, n_feature = X.shape
    c_class = len(np.unique(y))
    theta = np.random.rand(n_feature, c_class)
    one_hot = one_hot_encode(np.array(y))
    iter_cost = []
    for i in range(n_iter):
        pred = softmax_prediction(X, theta)
        change = np.dot(X.T, (pred - one_hot))
        theta = theta - alpha * np.array(change)
        cost = 0
        for c in range(c_class):
            cost += np.sum(one_hot[:, c] * np.log10(pred[:, c]))
        cost = -cost / m_sample
        iter_cost.append(cost)
    iter_list = [num for num in range(len(iter_cost))]
    plt.figure(figsize=(8, 6))
    plt.plot(iter_list, iter_cost, color='red')
    plt.suptitle('Gradient Method')
    plt.title('Cost Per Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.figtext(0.5, 0.01, f'Finished With Cost {iter_cost[-1].round(5)} In {i + 1} Iterations', ha='center')
    plt.show()
    return theta, iter_cost[-1].round(2)


# Naive Bayes Helpers
def convert(info, binary):
    m_sample = len(info) // 28
    data = []
    for i in range(m_sample):
        str_digit = info[i * 28: (i + 1) * 28]
        int_digit = np.zeros((28, 28), int)
        for j in range(28):
            for k in range(28):
                if binary:
                    if str_digit[j][k] != ' ':
                        int_digit[j][k] = 1
                else:
                    if str_digit[j][k] == ' ':
                        int_digit[j][k] = 0
                    elif str_digit[j][k] == '+':
                        int_digit[j][k] = 1
                    else:
                        int_digit[j][k] = 2
        data.append(int_digit)
    return data


def open_naive_bayes(binary):
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
    train_data = convert(train_info, binary)
    test_data = convert(test_info, binary)
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)


def class_probability(label):
    m_sample = len(label)
    _, count = np.unique(label, return_counts=True)
    class_prob = count / m_sample
    return class_prob


def index_probability(data, label, binary):
    unique, count = np.unique(label, return_counts=True)
    c_class = len(unique)
    m_sample = len(label)
    n_feature = 28 * 28
    if binary:
        index_prob = np.zeros((n_feature, c_class, 2))
    else:
        index_prob = np.zeros((n_feature, c_class, 3))
    for m in range(m_sample):
        for i in range(28):
            for j in range(28):
                if binary:
                    if data[m][i][j] == 0:
                        index_prob[28 * i + j][label[m]][0] += 1
                    elif data[m][i][j] == 1:
                        index_prob[28 * i + j][label[m]][1] += 1
                else:
                    if data[m][i][j] == 0:
                        index_prob[28 * i + j][label[m]][0] += 1
                    elif data[m][i][j] == 1:
                        index_prob[28 * i + j][label[m]][1] += 1
                    else:
                        index_prob[28 * i + j][label[m]][2] += 1
    for n in range(n_feature):
        for c in range(c_class):
            index_prob[n][c] = index_prob[n][c] / count[c]
    return index_prob


def naive_bayes_prediction(data, c_prob, i_prob, binary):
    c_class = len(c_prob)
    m_sample = len(data)
    data_pred = np.zeros((m_sample, 1), int)
    for i, d in enumerate(data):
        pred_arr = c_prob.copy()
        for j in range(28):
            for k in range(28):
                for m in range(c_class):
                    if binary:
                        if d[j][k] == 0:
                            pred_arr[m] *= i_prob[28 * j + k][m][0]
                        elif d[j][k] == 1:
                            pred_arr[m] *= i_prob[28 * j + k][m][1]
                    else:
                        if d[j][k] == 0:
                            pred_arr[m] *= i_prob[28 * j + k][m][0]
                        elif d[j][k] == 1:
                            pred_arr[m] *= i_prob[28 * j + k][m][1]
                        else:
                            pred_arr[m] *= i_prob[28 * j + k][m][2]
        data_pred[i][0] = np.argmax(pred_arr)
    return data_pred


def calc_scores(conf_mat):
    c_class = len(conf_mat)
    tp, tn, fp, fn = np.zeros(c_class, int), np.zeros(c_class, int), np.zeros(c_class, int), np.zeros(c_class, int)
    scores = np.zeros((c_class, 4))
    for i in range(c_class):
        tp[i] = conf_mat[i][i]
        tn[i] = np.sum(np.delete(np.delete(conf_mat, i, 0), i, 1))
        fp[i] = np.sum(np.delete(conf_mat[i, :], i))
        fn[i] = np.sum(np.delete(conf_mat[:, i], i, 0))
        scores[i][0] = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i])
        scores[i][1] = tp[i] / (tp[i] + fp[i])
        scores[i][2] = tp[i] / (tp[i] + fn[i])
        scores[i][3] = (2 * tp[i]) / ((2 * tp[i]) + fp[i] + fn[i])
    return scores


def confusion_score_matrix(label, pred):
    unique = np.unique(label)
    c_class = len(unique)
    label_index, pred_index = [], []
    conf_mat = np.zeros((c_class, c_class), int)
    for i in range(c_class):
        label_index.append(np.where(label == i)[0])
        pred_index.append(np.where(pred == i)[0])
    for i in range(c_class):
        for j in range(c_class):
            conf_mat[i][j] = len(np.intersect1d(pred_index[i], label_index[j]))
    score_mat = calc_scores(conf_mat)

    class_name = []
    for c in list(map(str, unique)):
        class_name.append('C' + c)

    conf_mat = pd.DataFrame(conf_mat, index=class_name, columns=class_name)
    score_mat = pd.DataFrame(score_mat, index=class_name, columns=['Accuracy', 'Precision', 'Recall', 'F1'])
    return conf_mat, score_mat
