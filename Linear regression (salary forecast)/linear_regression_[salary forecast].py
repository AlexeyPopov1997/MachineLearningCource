import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


# Function for text preprocessing
def text_transform(text: pandas.Series) -> pandas.Series:
    return text.str.lower().replace("[^a-zA-Z0-9]", " ", regex=True)


def get_train_data(train_file_name, test_file_name):
    train = pandas.read_csv(train_file_name)

    features_vector = TfidfVectorizer(min_df=5)
    features_train_text = features_vector.fit_transform(text_transform(train['FullDescription']))

    # Replacing blanks in the Location Normalized and Contract Time columns
    train['LocationNormalized'].fillna('nan', inplace=True)
    train['ContractTime'].fillna('nan', inplace=True)
    one_hot_coding_features = DictVectorizer()
    features_train_cat = one_hot_coding_features.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

    # Combining all the received features in one matrix "objects-features"
    features_train = hstack([features_train_text, features_train_cat])
    target_train = train['SalaryNormalized']

    test = pandas.read_csv(test_file_name)
    features_test_text = features_vector.transform(text_transform(test['FullDescription']))
    features_test_cat = one_hot_coding_features.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
    features_test = hstack([features_test_text, features_test_cat])

    return features_train, target_train, features_test


def main():
    x_train, y_train, x_test = get_train_data('salary-train.csv', 'salary-test-mini.csv')

    # Ridge regression training with parameters alpha = 1 and random_state = 241
    model = Ridge(alpha=1, random_state=241)
    model.fit(x_train, y_train)

    # Making forecasts for two examples from the salary-test-mini.csv file
    y_test = model.predict(x_test)
    print('Predicted values:', str(y_test[0].round(2)), str(y_test[1].round(2)))
    write_answer('1.txt', str(y_test[0].round(2)) + ' ' + str(y_test[1].round(2)))


if __name__ == '__main__':
    main()
