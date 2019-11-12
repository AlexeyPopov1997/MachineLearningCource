import matplotlib.pyplot as plt
import numpy
from pandas import read_csv
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from typing import List


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


# Function to obtain the necessary features
def get_data(file_name):
    data = read_csv(file_name)
    data.head()
    x = data.loc[:, 'D1':'D1776'].values
    y = data['Activity'].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8, random_state=241)
    return x_train, x_test, y_train, y_test


# Sigmoid function for predicting the quality of the training and test samples at each iteration
def sigmoid(y_prev: numpy.array) -> numpy.array:
    return 1.0 / (1.0 + numpy.exp(-y_prev))


# Function for predicting quality in training and test samples at each iteration
def log_loss_results(model, x: numpy.array, y: numpy.array) -> List[float]:
    return [log_loss(y, sigmoid(y_prev)) for y_prev in model.staged_decision_function(x)]


# Function for plotting log-loss values on training and test samples
def plot_loss(learning_rate: float, test_loss: List[float], train_loss: List[float]) -> None:
    plt.figure()
    plt.plot(test_loss, 'r', linewidth=2)
    plt.plot(train_loss, 'g', linewidth=2)
    plt.legend(['test', 'train'])
    plt.show()


# Function for finding the minimum metric value and the iteration number at which it is reached
def define_min_metric_value(x_train, x_test, y_train, y_test):
    min_loss_results = {}
    for lr in [1, 0.5, 0.3, 0.2, 0.1]:
        print('Learning rate:', str(lr))

        model = GradientBoostingClassifier(learning_rate=lr, n_estimators=250, verbose=True, random_state=241)
        model.fit(x_train, y_train)

        train_loss = log_loss_results(model, x_train, y_train)
        test_loss = log_loss_results(model, x_test, y_test)
        plot_loss(lr, test_loss, train_loss)

        min_loss_value = min(test_loss)
        min_loss_index = test_loss.index(min_loss_value) + 1
        min_loss_results[lr] = min_loss_value, min_loss_index

    min_loss_value, min_loss_index = min_loss_results[0.2]
    print('Min loss', str(min_loss_value.round(2)), 'at n_estimators =', str(min_loss_index), '\n')
    return min_loss_value, min_loss_index


def main():
    x_train, x_test, y_train, y_test = get_data('gbm-data.csv')
    # Getting the minimum value of log-loss on the test sample and the iteration number
    # at which it is achieved with learning_rate = 0.2
    min_loss_value, min_loss_index = define_min_metric_value(x_train, x_test, y_train, y_test)

    # On the same data we train RandomForestClassifier
    model = RandomForestClassifier(n_estimators=min_loss_index, random_state=241)
    model.fit(x_train, y_train)
    y_prev = model.predict_proba(x_test)[:, 1]
    test_loss = log_loss(y_test, y_prev)

    write_answer('1.txt', 'overfitting')
    print('The characterization the quality graph on a test sample, starting with some iteration:', 'overfitting')

    write_answer('2.txt', str(min_loss_value.round(2)) + str(' ') + str(min_loss_index))
    print('The minimum value of log-loss on the test sample and the iteration '
          'number at which it is achieved with learning_rate = 0.2:', str(min_loss_value.round(2))
          + str(' ') + str(min_loss_index))

    write_answer('3.txt', str(test_loss.round(2)))
    print('The quality on the test is obtained from a random forest from the fifth point:', str(test_loss.round(2)))


if __name__ == '__main__':
    main()
