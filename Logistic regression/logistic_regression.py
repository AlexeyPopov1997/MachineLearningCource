import pandas
import numpy
from sklearn.metrics import roc_auc_score
from typing import Tuple


# Function to obtain the necessary features and target function
def get_boston_data():
    data = pandas.read_csv('data-logistic.csv', header=None)
    features = data.loc[:, 1:]
    target = data[0]
    return features, target


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


# Function to update weight_1
def calc_weight_1(x: pandas.DataFrame, y: pandas.Series, weight_1: float, weight_2: float, k: float, c: float) -> float:
    sum = 0
    for i in range(0, len(y)):
        sum += y[i] * x[1][i] * (1.0 - 1.0 / (1.0 + numpy.exp(-y[i] * (weight_1*x[1][i] + weight_2*x[2][i]))))
    return weight_1 + (k * (1.0 / len(y)) * sum) - k * c * weight_1


# Function to update weight_2
def calc_weight_2(x: pandas.DataFrame, y: pandas.Series, weight_1: float, weight_2: float, k: float, C: float) -> float:
    sum = 0
    for i in range(0, len(y)):
        sum += y[i] * x[2][i] * (1.0 - 1.0 / (1.0 + numpy.exp(-y[i] * (weight_1*x[1][i] + weight_2*x[2][i]))))
    return weight_2 + (k * (1.0 / len(y)) * sum) - k * C * weight_2


# Implementation of gradient descent for ordinary and L2-regularized
# (with a regularization coefficient of 10) logistic regression
def gradient_descent(x: pandas.DataFrame, y: pandas.Series, weight_1: float = 0.0, weight_2: float = 0.0,
                     k: float = 0.1, c: float = 0.0, precision: float = 1e-5, max_iter: int = 10000) -> \
        Tuple[float, float]:
    for i in range(max_iter):
        weight_1_prev, weight_2_prev = weight_1, weight_2
        weight_1, weight_2 = calc_weight_1(x, y, weight_1, weight_2, k, c), calc_weight_2(x, y, weight_1, weight_2, k, c)
        if numpy.sqrt((weight_1_prev - weight_1) ** 2 + (weight_2_prev - weight_2) ** 2) <= precision:
            break
    return weight_1, weight_2


# Function for determining the value of AUC-ROC
def define_roc_auc_score(x: pandas.DataFrame, weight_1: float, weight_2: float) -> pandas.Series:
    return 1.0 / (1.0 + numpy.exp(-weight_1 * x[1] - weight_2 * x[2]))


def main():
    x, y = get_boston_data()
    # Starting gradient descent and bringing it to convergence
    weight_1, weight_2 = gradient_descent(x, y)
    weight_1_reg, weight_2_reg = gradient_descent(x, y, c=10.0)

    # Determination of AUC-ROCK value in training and regularization
    y_train = define_roc_auc_score(x, weight_1, weight_2)
    y_reg = define_roc_auc_score(x, weight_1_reg, weight_2_reg)
    auc_train = roc_auc_score(y, y_train)
    auc_reg = roc_auc_score(y, y_reg)

    write_answer('1.txt', str(auc_train.round(3)) + str(' ') + str(auc_reg.round(3)))
    print('AUC-ROC in learning without regularization and when using it:',
          str(auc_train.round(3)) + str(' ') + str(auc_reg.round(3)))


if __name__ == '__main__':
    main()
