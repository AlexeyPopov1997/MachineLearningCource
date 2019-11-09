import numpy
from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC


# Function to obtain the necessary features and target function
def get_data():
    newsgroups = datasets.fetch_20newsgroups(subset="all", categories=["alt.atheism", "sci.space"])
    # Calculation of TF-IDF tags for all texts
    vec = TfidfVectorizer()
    newsgroups_train = vec.fit_transform(newsgroups.data)
    return vec, newsgroups_train, newsgroups.target


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


def main():
    vec, x, y = get_data()

    grid = {'C': numpy.power(10.0, numpy.arange(-5, 6))}
    clf_cv_temp = KFold(n_splits=5, shuffle=True, random_state=241)
    clf_cv = clf_cv_temp.get_n_splits(len(y))
    clf_svc = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf_svc, grid, scoring='accuracy', cv=clf_cv)
    gs.fit(x, y)

    best_c = gs.get_params()['estimator__C']

    clf_svc = SVC(C=best_c, kernel='linear', random_state=241)
    clf_svc.fit(x, y)

    # Finding the 10 words with the highest modulus weight
    most_important_words_indexes = numpy.argsort(abs(clf_svc.coef_.toarray()[0]))[-10:]
    most_important_words = numpy.array(vec.get_feature_names())[most_important_words_indexes]
    most_important_words_sorted = sorted(most_important_words)
    resulting_string = ','.join(most_important_words_sorted)

    write_answer('1.txt', resulting_string)
    print('10 words with the largest modulus weight:', resulting_string)


if __name__ == '__main__':
    main()
