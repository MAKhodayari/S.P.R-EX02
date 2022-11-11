import numpy as np
import utilities as utl
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Opening & Preparing Data
    data = utl.open_logistic()

    # One VS One
    data1v2 = data.drop(data[data.y == 3].index)
    data1v3 = data.drop(data[data.y == 2].index)
    data2v3 = data.drop(data[data.y == 1].index)

    X1v2, y1v2 = data1v2.iloc[:, :-1], data1v2.iloc[:, -1]
    X1v3, y1v3 = data1v3.iloc[:, :-1], data1v3.iloc[:, -1]
    X2v3, y2v3 = data2v3.iloc[:, :-1], data2v3.iloc[:, -1]

    X1v2 = utl.normalize(X1v2)
    X1v3 = utl.normalize(X1v3)
    X2v3 = utl.normalize(X2v3)
    X1v2.insert(0, 'X0', 1)
    X1v3.insert(0, 'X0', 1)
    X2v3.insert(0, 'X0', 1)

    y1v2 = np.array([0 if y == 1 else 1 for y in y1v2]).reshape(-1, 1)
    y1v3 = np.array([0 if y == 1 else 1 for y in y1v3]).reshape(-1, 1)
    y2v3 = np.array([0 if y == 2 else 1 for y in y2v3]).reshape(-1, 1)

    X1v2_train, X1v2_test, y1v2_train, y1v2_test = train_test_split(X1v2, y1v2, test_size=0.2)
    X1v3_train, X1v3_test, y1v3_train, y1v3_test = train_test_split(X1v3, y1v3, test_size=0.2)
    X2v3_train, X2v3_test, y2v3_train, y2v3_test = train_test_split(X2v3, y2v3, test_size=0.2)

    theta1v2 = utl.logistic_gradient(X1v2_train, y1v2_train, 0.25, 5000)
    theta1v3 = utl.logistic_gradient(X1v3_train, y1v3_train, 0.25, 5000)
    theta2v3 = utl.logistic_gradient(X2v3_train, y2v3_train, 0.25, 5000)

    ce_train_1v2 = utl.calc_cross_entropy(X1v2_train, y1v2_train, theta1v2)
    ce_test_1v2 = utl.calc_cross_entropy(X1v2_test, y1v2_test, theta1v2)

    yh_train_1v2 = utl.logistic_prediction(X1v2_train, theta1v2, norm=True)
    yh_test_1v2 = utl.logistic_prediction(X1v2_test, theta1v2, norm=True)

    train_acc_1v2 = utl.calc_accuracy(y1v2_train, yh_train_1v2)
    test_acc_1v2 = utl.calc_accuracy(y1v2_test, yh_test_1v2)

    print(f'Theta: {theta1v2}')
    print(f'Train Cross Entropy: {ce_train_1v2} | Test Cross Entropy: {ce_test_1v2}')
    print(f'Train Accuracy {train_acc_1v2} | Test Accuracy {test_acc_1v2}')

    ce_train_1v3 = utl.calc_cross_entropy(X1v3_train, y1v3_train, theta1v3)
    ce_test_1v3 = utl.calc_cross_entropy(X1v3_test, y1v3_test, theta1v3)

    yh_train_1v3 = utl.logistic_prediction(X1v3_train, theta1v3, norm=True)
    yh_test_1v3 = utl.logistic_prediction(X1v3_test, theta1v3, norm=True)

    train_acc_1v3 = utl.calc_accuracy(y1v3_train, yh_train_1v3)
    test_acc_1v3 = utl.calc_accuracy(y1v3_test, yh_test_1v3)

    print(f'Theta: {theta1v3}')
    print(f'Train Cross Entropy: {ce_train_1v3} | Test Cross Entropy: {ce_test_1v3}')
    print(f'Train Accuracy {train_acc_1v3} | Test Accuracy {test_acc_1v3}')

    ce_train_2v3 = utl.calc_cross_entropy(X2v3_train, y2v3_train, theta2v3)
    ce_test_2v3 = utl.calc_cross_entropy(X2v3_test, y2v3_test, theta2v3)

    yh_train_2v3 = utl.logistic_prediction(X2v3_train, theta2v3, norm=True)
    yh_test_2v3 = utl.logistic_prediction(X2v3_test, theta2v3, norm=True)

    train_acc_2v3 = utl.calc_accuracy(y2v3_train, yh_train_2v3)
    test_acc_2v3 = utl.calc_accuracy(y2v3_test, yh_test_2v3)

    print(f'Theta: {theta2v3}')
    print(f'Train Cross Entropy: {ce_train_2v3} | Test Cross Entropy: {ce_test_2v3}')
    print(f'Train Accuracy {train_acc_2v3} | Test Accuracy {test_acc_2v3}')
