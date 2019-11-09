import pandas


# Determination of the number of men on the ship
def define_males_count(data, col_name):
    males_count = data[data.Sex == 'male'][col_name].count()
    print('Males count: ', males_count)
    return males_count


# Determination of the number of women on the ship
def define_females_count(data, col_name):
    females_count = data[data.Sex == 'female'][col_name].count()
    print('Females count: ', females_count)
    return females_count


# Determination of the survived
# If the value of roundValue = None, displays the number of survivors,
# if not, then the percentage of survivors is displayed, rounded in accordance with the value of roundValue
def define_survived(data, col_name, round_value=None):
    if round_value is not None:
        survived_count = data[data.Survived == 1][col_name].count()
        total_passengers_count = data[col_name].count()
        survived_count_percent = 100 * survived_count / total_passengers_count
        print('Count of survived (percent): ', survived_count_percent.round(round_value))
        return survived_count_percent.round(round_value)
    else:
        survived_count = data[data.Survived == 1][col_name].count()
        print('Count of survived: ', survived_count)
        return survived_count


# Passenger class definition
# If the value roundValue = None,
# the number of passengers of a certain class (specified using the classValue parameter) is displayed,
# if not, the percentage of passengers is displayed, a specific class, rounded according to the value of roundValue
def define_class_passengers(data, col_name, class_value, round_value=None):
    if round_value is not None:
        first_class_passengers_count = data[data.Pclass == class_value][col_name].count()
        total_passengers_count = data[col_name].count()
        first_class_passengers_count_percent = 100 * first_class_passengers_count / total_passengers_count
        print(class_value, 'st class passengers (percent): ', first_class_passengers_count_percent.round(round_value))
        return first_class_passengers_count_percent.round(round_value)
    else:
        first_class_passengers_count = data[data.Pclass == class_value][col_name].count()
        print(class_value, 'st class passengers: ', first_class_passengers_count)
        return first_class_passengers_count


# Determining the average age of passengers
def define_average_age(data, col_name, round_value=None):
    if round_value is not None:
        average_age = round(data[col_name].mean(), round_value)
        print('Average age of passengers:', average_age)
        return average_age
    else:
        average_age = round(data[col_name].mean(), 2)
        print('Average age of passengers:', average_age)
        return average_age


# Determining the median age of passengers
def define_median_age(data, col_name, round_value=None):
    if round_value is not None:
        median_age = round(data[col_name].median(), round_value)
        print('Median age of passengers:', median_age)
        return median_age
    else:
        median_age = round(data[col_name].median(), 2)
        print('Median age of passengers:', median_age)
        return median_age


# Counting the correlation value between two columns
def calculate_correlation(data, first_col, second_col, method_name, round_value=None):
    if round_value is not None:
        corr_value = round(data[first_col].corr(data[second_col], method=method_name), round_value)
        print('Pearson correlation value:', corr_value)
        return corr_value
    else:
        corr_value = round(data[first_col].corr(data[second_col], method=method_name), 2)
        print('Pearson correlation value:', round(data[first_col].corr(data[second_col], method=method_name), 2))
        return corr_value


# Identify the most popular name
def define_most_popular_name(data, col_name, sex):
    female_names = data[data.Sex == sex][col_name]
    temp = []
    for index in female_names:
        if '(' in index:
            if ')' in index.split('(')[1].split(' ')[0]:
                temp.append(index.split('(')[1].split(' ')[0].split(')')[0])
            else:
                temp.append(index.split('(')[1].split(' ')[0])
        else:
            temp.append(index.split('. ')[1].split(' ')[0])
    most_popular_name = str(pandas.DataFrame.from_dict(temp)[0].value_counts()[:1])
    index = 0
    while most_popular_name[index] != ' ':
        index = index + 1
    if sex == 'female':
        string = 'The most popular female name:'
    else:
        string = 'The most popular male name:'
    print(string, most_popular_name[:index])
    return most_popular_name[:index]


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


def main():
    data = pandas.read_csv('titanic.csv', index_col='PassengerId')
    write_answer('1.txt', str(define_males_count(data, 'Sex')) + ' ' + str(define_females_count(data, 'Sex')))
    write_answer('2.txt', str(define_survived(data, 'Survived', 2)))
    write_answer('3.txt', str(define_class_passengers(data, 'Pclass', 1, 2)))
    write_answer('4.txt', str(define_average_age(data, 'Age', 2)) + ' ' + str(define_median_age(data, 'Age', 2)))
    write_answer('5.txt', str(calculate_correlation(data, 'SibSp', 'Parch', 'pearson')))
    write_answer('6.txt', str(define_most_popular_name(data, 'Name', 'female')))


if __name__ == '__main__':
    main()
