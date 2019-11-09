import pandas
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale


# Function to obtain the necessary features and target function
def get_data(file_name, target_col_name):
    # Column names taken from file wine.names
    columns = [
        'Class',
        'Alcohol',
        'Malic acid',
        'Ash',
        'Alcalinity of ash',
        'Magnesium',
        'Total phenols',
        'Flavanoids',
        'Nonflavanoid phenols',
        'Proanthocyanins',
        'Color intensity',
        'Hue',
        'OD280/OD315 of diluted wines',
        'Proline',
    ]
    data = pandas.read_csv(file_name, index_col=False, names=columns)
    features = data.loc[:, data.columns != 'Class']
    target = data[target_col_name]
    return features, target


# Function for determining the accuracy of cross-validation classification for the k nearest neighbors method.
# Returns the value of the classification accuracy and the value of k at which this accuracy is achieved
def define_cross_val_clf_accuracy(features, target, cv):
    best_scoring, best_k = None, None

    for k in range(1, 51):
        model = KNeighborsClassifier(n_neighbors=k)
        scoring = cross_val_score(model, features, target, cv=cv, scoring='accuracy').mean()

        if best_scoring is None or scoring > best_scoring:
            best_scoring, best_k = scoring, k

    return best_scoring, best_k


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


def main():
    x, y = get_data('wine.data', 'Class')
    clf = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring, k = define_cross_val_clf_accuracy(x, y, clf)
    write_answer('1.txt', str(k))
    print('k at which optimum accuracy is achieved:', str(k))
    write_answer('2.txt', str(scoring.round(2)))
    print('Value at which optimum classification accuracy is achieved:', str(scoring.round(2)))
    scoring, k = define_cross_val_clf_accuracy(scale(x), y, clf)
    write_answer('3.txt', str(k))
    print('k at which optimum accuracy is achieved after normalizing features:', str(k))
    write_answer('4.txt', str(scoring.round(2)))
    print('Value at which optimum classification accuracy is achieved after normalizing features:',
          str(scoring.round(2)))


if __name__ == '__main__':
    main()



