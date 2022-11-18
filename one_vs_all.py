import numpy as np
import utilities as utl
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # Opening Dataset
    data = utl.open_logistic()

    # Preparing Data For One VS All Method
    X_OvA = data.iloc[:, :-1]

    y_1vA = np.array([1 if label == 1 else 0 for label in data.y]).reshape(-1, 1)
    y_2vA = np.array([1 if label == 2 else 0 for label in data.y]).reshape(-1, 1)
    y_3vA = np.array([1 if label == 3 else 0 for label in data.y]).reshape(-1, 1)

    X_OvA = utl.normalize(X_OvA)
    X_OvA.insert(0, 'X0', 1)

    X_train_1vA, X_test_1vA, y_train_1vA, y_test_1vA = train_test_split(X_OvA, y_1vA, test_size=0.2)
    X_train_2vA, X_test_2vA, y_train_2vA, y_test_2vA = train_test_split(X_OvA, y_2vA, test_size=0.2)
    X_train_3vA, X_test_3vA, y_train_3vA, y_test_3vA = train_test_split(X_OvA, y_3vA, test_size=0.2)

    # Train Phase Of 1 VS All
    theta_1vA = utl.logistic_gradient(X_train_1vA, y_train_1vA, 0.5, 10000)

    # Test Phase Of 1 VS All
    yh_train_1vA = utl.logistic_prediction(X_train_1vA, theta_1vA, 1 / 3, True)
    yh_test_1vA = utl.logistic_prediction(X_test_1vA, theta_1vA, 1 / 3, True)

    train_ce_1vA = utl.calc_cross_entropy(X_train_1vA, y_train_1vA, theta_1vA)
    test_ce_1vA = utl.calc_cross_entropy(X_test_1vA, y_test_1vA, theta_1vA)

    train_acc_1vA = utl.calc_accuracy(y_train_1vA, yh_train_1vA)
    test_acc_1vA = utl.calc_accuracy(y_test_1vA, yh_test_1vA)

    # Results For 1 VS All
    print('1 VS All:')
    print(f'Theta: {theta_1vA.round(2)}')
    print(f'Train Cross Entropy: {train_ce_1vA.round(2)} | Test Cross Entropy: {test_ce_1vA.round(2)}')
    print(f'Train Accuracy: {round(train_acc_1vA * 100, 2)} | Test Accuracy: {round(test_acc_1vA * 100, 2)}\n')

    # Train Phase Of 2 VS All
    theta_2vA = utl.logistic_gradient(X_train_2vA, y_train_2vA, 0.5, 10000)

    # Test Phase Of 2 VS All
    yh_train_2vA = utl.logistic_prediction(X_train_2vA, theta_2vA, 1 / 3, True)
    yh_test_2vA = utl.logistic_prediction(X_test_2vA, theta_2vA, 1 / 3, True)

    train_ce_2vA = utl.calc_cross_entropy(X_train_2vA, y_train_2vA, theta_2vA)
    test_ce_2vA = utl.calc_cross_entropy(X_test_2vA, y_test_2vA, theta_2vA)

    train_acc_2vA = utl.calc_accuracy(y_train_2vA, yh_train_2vA)
    test_acc_2vA = utl.calc_accuracy(y_test_2vA, yh_test_2vA)

    # Results For 2 VS All
    print('2 VS All:')
    print(f'Theta: {theta_2vA.round(2)}')
    print(f'Train Cross Entropy: {train_ce_2vA.round(2)} | Test Cross Entropy: {test_ce_2vA.round(2)}')
    print(f'Train Accuracy: {round(train_acc_2vA * 100, 2)} | Test Accuracy: {round(test_acc_2vA * 100, 2)}\n')

    # Train Phase Of 3 VS All
    theta_3vA = utl.logistic_gradient(X_train_3vA, y_train_3vA, 0.5, 10000)

    # Test Phase Of 2 VS All
    yh_train_3vA = utl.logistic_prediction(X_train_3vA, theta_3vA, 1 / 3, True)
    yh_test_3vA = utl.logistic_prediction(X_test_3vA, theta_3vA, 1 / 3, True)

    train_ce_3vA = utl.calc_cross_entropy(X_train_3vA, y_train_3vA, theta_3vA)
    test_ce_3vA = utl.calc_cross_entropy(X_test_3vA, y_test_3vA, theta_3vA)

    train_acc_3vA = utl.calc_accuracy(y_train_3vA, yh_train_3vA)
    test_acc_3vA = utl.calc_accuracy(y_test_3vA, yh_test_3vA)

    # Results For 3 VS All
    print('3 VS All:')
    print(f'Theta: {theta_3vA.round(2)}')
    print(f'Train Cross Entropy: {train_ce_3vA.round(2)} | Test Cross Entropy: {test_ce_3vA.round(2)}')
    print(f'Train Accuracy: {round(train_acc_3vA * 100, 2)} | Test Accuracy: {round(test_acc_3vA * 100, 2)}\n')
