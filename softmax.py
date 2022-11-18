import numpy as np
import utilities as utl
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Opening & Preparing Data
    data = utl.open_logistic()

    X, y = data.iloc[:, :-1], data.iloc[:, -1]

    X = utl.normalize(X)
    X.insert(0, 'X0', 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train
    class_theta, cross_entropy = utl.softmax_gradient(X_train, y_train, 0.01, 1000)

    # Test
    yh_train = utl.softmax_prediction(X_train, class_theta, True)
    yh_test = utl.softmax_prediction(X_test, class_theta, True)

    acc_train = utl.calc_accuracy(np.array(y_train).reshape(-1, 1), yh_train)
    acc_test = utl.calc_accuracy(np.array(y_test).reshape(-1, 1), yh_test)

    # Results
    print('Softmax Regression Result:')
    for c in range(len(np.unique(y))):
        print(f'Theta For Class {c + 1}: {class_theta[:, c].round(2)}')
    print(f'Cross Entropy: {cross_entropy}')
    print(f'Train Accuracy: {round(acc_train * 100, 2)} | Test Accuracy: {round(acc_test * 100, 2)}\n')
