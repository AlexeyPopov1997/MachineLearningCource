from numpy import linspace
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor


# Function to obtain the necessary features and target function
def get_boston_data():
    boston_data = load_boston()
    features = boston_data.data
    target = boston_data.target
    return features, target


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


def main():
    x, y = get_boston_data()
    x_scaled = scale(x)
    metric_parameters = linspace(1.0, 10.0, num=200)
    clf_temp = KFold(n_splits=5, shuffle=True, random_state=42)
    clf = clf_temp.get_n_splits(len(x_scaled))
    accuracy_cv = [cross_val_score(estimator=KNeighborsRegressor(n_neighbors=5, weights='distance', p=p_i,
                                                                 metric='minkowski'),
                                   X=x_scaled, y=y, cv=clf).mean() for p_i in metric_parameters]
    best_metric_parameters = metric_parameters[int(max(accuracy_cv))]
    write_answer('1.txt', str(best_metric_parameters))
    print('For p = ', best_metric_parameters, ', the accuracy of cross-validation turned out to be optimal.', sep='')


if __name__ == '__main__':
    main()
