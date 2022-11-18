import numpy as np
import utilities as utl
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Opening Dataset
    data = utl.open_logistic()

    # Preparing Data For One VS One Method
    data_1v2 = data.drop(data[data.y == 3].index)
    data_1v3 = data.drop(data[data.y == 2].index)
    data_2v3 = data.drop(data[data.y == 1].index)

    X_1v2, y_1v2 = data_1v2.iloc[:, :-1], data_1v2.iloc[:, -1]
    X_1v3, y_1v3 = data_1v3.iloc[:, :-1], data_1v3.iloc[:, -1]
    X_2v3, y_2v3 = data_2v3.iloc[:, :-1], data_2v3.iloc[:, -1]

    X_1v2 = utl.normalize(X_1v2)
    X_1v3 = utl.normalize(X_1v3)
    X_2v3 = utl.normalize(X_2v3)
    X_1v2.insert(0, 'X0', 1)
    X_1v3.insert(0, 'X0', 1)
    X_2v3.insert(0, 'X0', 1)

    y_1v2 = np.array([0 if y == 1 else 1 for y in y_1v2]).reshape(-1, 1)
    y_1v3 = np.array([0 if y == 1 else 1 for y in y_1v3]).reshape(-1, 1)
    y_2v3 = np.array([0 if y == 2 else 1 for y in y_2v3]).reshape(-1, 1)

    X_train_1v2, X_test_1v2, y_train_1v2, y_test_1v2 = train_test_split(X_1v2, y_1v2, test_size=0.2)
    X_train_1v3, X_test_1v3, y_train_1v3, y_test_1v3 = train_test_split(X_1v3, y_1v3, test_size=0.2)
    X_train_2v3, X_test_2v3, y_train_2v3, y_test_2v3 = train_test_split(X_2v3, y_2v3, test_size=0.2)

    # Train Phase Of 1 VS 2
    theta_1v2 = utl.logistic_gradient(X_train_1v2, y_train_1v2, 0.1, 5000)

    # Test Phase Of 1 VS 2
    yh_train_1v2 = utl.logistic_prediction(X_train_1v2, theta_1v2, norm=True)
    yh_test_1v2 = utl.logistic_prediction(X_test_1v2, theta_1v2, norm=True)

    train_ce_1v2 = utl.calc_cross_entropy(X_train_1v2, y_train_1v2, theta_1v2)
    test_ce_1v2 = utl.calc_cross_entropy(X_test_1v2, y_test_1v2, theta_1v2)

    train_acc_1v2 = utl.calc_accuracy(y_train_1v2, yh_train_1v2)
    test_acc_1v2 = utl.calc_accuracy(y_test_1v2, yh_test_1v2)

    # Results For 1 VS 2
    print('1 VS 2:')
    print(f'Theta: {theta_1v2.round(2)}')
    print(f'Train Cross Entropy: {train_ce_1v2.round(2)} | Test Cross Entropy: {test_ce_1v2.round(2)}')
    print(f'Train Accuracy: {round(train_acc_1v2 * 100, 2)} | Test Accuracy: {round(test_acc_1v2 * 100, 2)}\n')

    # Train Phase Of 1 VS 3
    theta_1v3 = utl.logistic_gradient(X_train_1v3, y_train_1v3, 0.1, 5000)

    # Test Phase Of 1 VS 3
    yh_train_1v3 = utl.logistic_prediction(X_train_1v3, theta_1v3, norm=True)
    yh_test_1v3 = utl.logistic_prediction(X_test_1v3, theta_1v3, norm=True)

    train_ce_1v3 = utl.calc_cross_entropy(X_train_1v3, y_train_1v3, theta_1v3)
    test_ce_1v3 = utl.calc_cross_entropy(X_test_1v3, y_test_1v3, theta_1v3)

    train_acc_1v3 = utl.calc_accuracy(y_train_1v3, yh_train_1v3)
    test_acc_1v3 = utl.calc_accuracy(y_test_1v3, yh_test_1v3)

    # Results For 1 VS 3
    print('1 VS 3:')
    print(f'Theta: {theta_1v3.round(2)}')
    print(f'Train Cross Entropy: {train_ce_1v3.round(2)} | Test Cross Entropy: {test_ce_1v3.round(2)}')
    print(f'Train Accuracy: {round(train_acc_1v3 * 100, 2)} | Test Accuracy: {round(test_acc_1v3 * 100, 2)}\n')

    # Train Phase Of 2 VS 3
    theta_2v3 = utl.logistic_gradient(X_train_2v3, y_train_2v3, 0.1, 5000)

    # Test Phase Of 2 VS 3
    yh_train_2v3 = utl.logistic_prediction(X_train_2v3, theta_2v3, norm=True)
    yh_test_2v3 = utl.logistic_prediction(X_test_2v3, theta_2v3, norm=True)

    train_ce_2v3 = utl.calc_cross_entropy(X_train_2v3, y_train_2v3, theta_2v3)
    test_ce_2v3 = utl.calc_cross_entropy(X_test_2v3, y_test_2v3, theta_2v3)

    train_acc_2v3 = utl.calc_accuracy(y_train_2v3, yh_train_2v3)
    test_acc_2v3 = utl.calc_accuracy(y_test_2v3, yh_test_2v3)

    # Results For 2 VS 3
    print('2 VS 3:')
    print(f'Theta: {theta_2v3.round(2)}')
    print(f'Train Cross Entropy: {train_ce_2v3.round(2)} | Test Cross Entropy: {test_ce_2v3.round(2)}')
    print(f'Train Accuracy: {round(train_acc_2v3 * 100, 2)} | Test Accuracy: {round(test_acc_2v3 * 100, 2)}\n')
