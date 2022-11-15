import utilities as utl


if __name__ == '__main__':
    # Opening & Preparing Data
    train_data, train_label, test_data, test_label = utl.open_naive_bayes()

    # Train
    class_prob = utl.class_probability(train_label)
    index_prob = utl.index_probability(train_data, train_label, 10 ** -7)

    # Test
    data_pred = utl.naive_bayes_prediction(test_data, class_prob, index_prob)
    naive_bayes_acc = utl.calc_accuracy(test_label, data_pred)

    # Result
    print(f'Naive Bayes Accuracy: {round(naive_bayes_acc, 2)}')
