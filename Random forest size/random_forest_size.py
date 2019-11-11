from pandas import read_csv
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


# Function to obtain the necessary features and target function
def get_data(file_name):
    data = read_csv(file_name)
    data['Sex'].replace({'F': -1, 'I': 0, 'M': 1}, inplace=True)
    features = data.loc[:, 'Sex':'ShellWeight']
    target = data['Rings']
    return features, target


# Function for training a random forest with a different number of trees and
# assessing the score of the resulting forest on cross-validation in 5 blocks
def define_scores(features, target, clf):
    scores = []
    for n in range(1, 51):
        model_cv = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1)
        score = cross_val_score(model_cv, features, target, cv=clf, scoring='r2').mean()
        scores.append(score)
    return scores


# Function for minimum number of trees at which a random forest shows
# the score of cross-validation above 0.52 definition
def define_min_count_trees(scores, result=None):
    for n, score in enumerate(scores):
        if score > 0.52:
            result = str(n + 1)
            break
    return result


def main():
    x, y = get_data('abalone.csv')
    clf = KFold(n_splits=5, shuffle=True, random_state=1)
    scores = define_scores(x, y, clf)
    min_num_trees = define_min_count_trees(scores)
    write_answer('1.txt', str(min_num_trees))
    print('The minimum number of trees at which a random forest shows the score of cross-validation above 0.52:',
          str(min_num_trees))


if __name__ == '__main__':
    main()
