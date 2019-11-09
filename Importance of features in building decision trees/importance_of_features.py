import pandas
import numpy
from sklearn.tree import DecisionTreeClassifier
from collections import Counter


# Function to obtain the necessary features and target function
def get_data():
    data = pandas.read_csv('titanic.csv', index_col='PassengerId')
    data['Sex'] = data['Sex'].factorize()[0]
    temp = data[~numpy.isnan(data['Age'])]
    features = temp[['Pclass', 'Fare', 'Age', 'Sex']]
    target = temp[['Survived']]
    return features, target


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


def main():
    x, y = get_data()
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(x, y)
    feature_importance = dict(zip(x, clf.feature_importances_))
    feature_importance = Counter(feature_importance)
    answer_string = feature_importance.most_common(1)[0][0] + ' ' + feature_importance.most_common(2)[1][0]
    write_answer('1.txt', answer_string)
    print('Features of greatest importance:', answer_string)


if __name__ == '__main__':
    main()
