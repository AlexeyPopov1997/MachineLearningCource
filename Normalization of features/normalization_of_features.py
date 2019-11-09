from pandas import read_csv
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# Function to obtain the necessary features and target function
def get_data(train_file_name, test_file_name):
    data_train = read_csv(train_file_name, header=None)
    data_test = read_csv(test_file_name, header=None)
    features_train = data_train.loc[:, 1:]
    target_test = data_train[0]
    features_train = data_train.loc[:, 1:]
    target_train = data_train[0]
    features_test = data_test.loc[:, 1:]
    target_test = data_test[0]
    return features_train, target_train, features_test, target_test


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


def main():
    x_train, y_train, x_test, y_test = get_data('perceptron-train.csv', 'perceptron-test.csv')
    clf = Perceptron(random_state=241, max_iter=5, tol=None)
    clf.fit(x_train, y_train)
    accuracy_before = accuracy_score(y_test, clf.predict(x_test))

    std_scl = StandardScaler()
    x_train_scaled = std_scl.fit_transform(x_train)
    x_test_scaled = std_scl.transform(x_test)
    clf.fit(x_train_scaled, y_train)
    accuracy_after = accuracy_score(y_test, clf.predict(x_test_scaled))

    difference = accuracy_after - accuracy_before
    write_answer('1.txt', str(difference.round(3)))
    print('The difference between the accuracy on the test sample after normalization and the accuracy before it:',
          str(difference.round(3)))


if __name__ == '__main__':
    main()
