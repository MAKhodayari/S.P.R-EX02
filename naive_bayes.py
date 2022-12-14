import utilities as utl


if __name__ == '__main__':
    # Opening & Preparing Data
    binary_train_data, binary_train_label, binary_test_data, binary_test_label = utl.open_naive_bayes(True)
    ternary_train_data, ternary_train_label, ternary_test_data, ternary_test_label = utl.open_naive_bayes(False)

    # Train
    binary_class_prob = utl.class_probability(binary_train_label)
    ternary_class_prob = utl.class_probability(ternary_train_label)

    binary_index_prob = utl.index_probability(binary_train_data, binary_train_label, 0.83, True)
    ternary_index_prob = utl.index_probability(ternary_train_data, ternary_train_label, 0.83, False)

    # Test
    binary_data_pred = utl.naive_bayes_prediction(binary_test_data, binary_class_prob, binary_index_prob, True)
    ternary_data_pred = utl.naive_bayes_prediction(ternary_test_data, ternary_class_prob, ternary_index_prob, False)

    binary_naive_bayes_acc = utl.calc_accuracy(binary_test_label, binary_data_pred)
    ternary_naive_bayes_acc = utl.calc_accuracy(ternary_test_label, ternary_data_pred)

    binary_conf_mat, binary_score_mat = utl.confusion_score_matrix(binary_test_label, binary_data_pred)
    ternary_conf_mat, ternary_score_mat = utl.confusion_score_matrix(ternary_test_label, ternary_data_pred)

    # Results
    print('Binary Confusion Matrix:')
    print(binary_conf_mat)

    print()

    print('Binary Score Matrix:')
    print(binary_score_mat.round(2))

    print()

    print('Ternary Confusion Matrix:')
    print(ternary_conf_mat)

    print()

    print('Ternary Score Matrix:')
    print(ternary_score_mat.round(2))

    print()

    print('Naive Bayes Accuracy With Binary Features: ' + str(round(binary_naive_bayes_acc * 100, 2)))
    print('Naive Bayes Accuracy With Ternary Features: ' + str(round(ternary_naive_bayes_acc * 100, 2)))
