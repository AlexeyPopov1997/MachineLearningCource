from pandas import read_csv
from sklearn.svm import SVC


# Function to obtain the necessary features and target function
def get_data(file_name):
    data = read_csv(file_name, index_col=None, header=None)
    features = data.loc[:, 1:].copy()
    target = data[0]
    return features, target


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


def main():
    x, y = get_data('svm-data.csv')
    clf = SVC(C=100000, kernel='linear', random_state=241)
    clf.fit(x, y)

    # Find the numbers of objects that are reference (numbering from one)
    # Indices of reference objects of the trained classifier are stored in the support_ field
    support_indexes = clf.support_

    # The numbering of reference objects begins with one
    str_support_indexes = [repr(i + 1) for i in support_indexes]
    resulting_string = ','.join(str_support_indexes)
    write_answer('1.txt', resulting_string)
    print('Reference object indexes numbers:', resulting_string)


if __name__ == '__main__':
    main()
