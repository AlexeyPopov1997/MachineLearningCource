import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')


# Display data frame
def show_data():
    print(data)


# Determination of the number of men on the ship
def define_males_count():
    males_count = data[data.Sex == 'male']['Sex'].count()
    print('Males count: ', males_count)
    return males_count


# Determination of the number of women on the ship
def define_females_count():
    females_count = data[data.Sex == 'female']['Sex'].count()
    print('Females count: ', females_count)
    return females_count


# Determination of the survived
# If the value of roundValue = None, displays the number of survivors,
# if not, then the percentage of survivors is displayed, rounded in accordance with the value of roundValue
def define_survived(round_value=None):
    if round_value is not None:
        survived_count = data[data.Survived == 1]['Survived'].count()
        total_passengers_count = data['Survived'].count()
        survived_count_percent = 100 * survived_count / total_passengers_count
        print('Count of survived (percent): ', survived_count_percent.round(round_value))
        return survived_count_percent.round(round_value)
    else:
        survived_count = data[data.Survived == 1]['Survived'].count()
        print('Count of survived: ', survived_count)
        return survived_count


# Passenger class definition
# If the value roundValue = None,
# the number of passengers of a certain class (specified using the classValue parameter) is displayed,
# if not, the percentage of passengers is displayed, a specific class, rounded according to the value of roundValue
def define_class_passengers(class_value, round_value=None):
    if round_value is not None:
        first_class_passengers_count = data[data.Pclass == class_value]['Pclass'].count()
        total_passengers_count = data['Pclass'].count()
        first_class_passengers_count_percent = 100 * first_class_passengers_count / total_passengers_count
        print(class_value, 'st class passengers (percent): ', first_class_passengers_count_percent.round(round_value))
        return first_class_passengers_count_percent.round(round_value)
    else:
        first_class_passengers_count = data[data.Pclass == class_value]['Pclass'].count()
        print(class_value, 'st class passengers: ', first_class_passengers_count)
        return first_class_passengers_count


# Determining the average age of passengers
def define_average_age(round_value=None):
    if round_value is not None:
        average_age = round(data['Age'].mean(), round_value)
        print('Average age of passengers:', average_age)
        return average_age
    else:
        average_age = round(data['Age'].mean(), 2)
        print('Average age of passengers:', average_age)
        return average_age


# Determining the median age of passengers
def define_median_age(round_value=None):
    if round_value is not None:
        median_age = round(data['Age'].median(), round_value)
        print('Median age of passengers:', median_age)
        return median_age
    else:
        median_age = round(data['Age'].median(), 2)
        print('Median age of passengers:', median_age)
        return median_age


# Counting the correlation value between two columns
def calculate_correlation(first_col, second_col, method_name, round_value=None):
    if round_value is not None:
        corr_value = round(data[first_col].corr(data[second_col], method=method_name), round_value)
        print('Pearson correlation value:', corr_value)
        return corr_value
    else:
        corr_value = round(data[first_col].corr(data[second_col], method=method_name), 2)
        print('Pearson correlation value:', round(data[first_col].corr(data[second_col], method=method_name), 2))
        return corr_value


# Identify the most popular name
def define_most_popular_name(sex):
    female_names = data[data.Sex == sex]['Name']
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


# Answers on questions

# Question 1
# How many men and women rode the ship? As an answer, give two numbers separated by a space
file_answer1 = open("answer # 1.txt", "w")
answer1 = str(define_males_count()) + ' ' + str(define_females_count())
file_answer1.write(answer1)
file_answer1.close()

# Question 2
# How many passengers survived?
# Calculate the proportion of surviving passengers.
# Give the answer in percent (a number in the range from 0 to 100, a percent sign is not needed),
# rounded to two characters
file_answer2 = open("answer # 2.txt", "w")
answer2 = str(define_survived(2))
file_answer2.write(answer2)
file_answer2.close()

# Question 3
# What is the proportion of first-class passengers among all passengers?
# Give the answer in percent (a number in the range from 0 to 100, a percent sign is not needed),
# rounded to two characters
file_answer3 = open("answer # 3.txt", "w")
answer3 = str(define_class_passengers(1, 2))
file_answer3.write(answer3)
file_answer3.close()

# Question 4
# How old were the passengers?
# Calculate the average and median age of passengers.
# As an answer, give two numbers separated by a space
file_answer4 = open("answer # 4.txt", "w")
answer4 = str(define_average_age(2)) + ' ' + str(define_median_age(2))
file_answer4.write(answer4)
file_answer4.close()

# Question 5
# Do the number of brothers / sisters / spouses correlate with the number of parents / children?
# Count the Pearson correlation between the SibSp and Parch traits
file_answer5 = open("answer # 5.txt", "w")
answer5 = str(calculate_correlation('SibSp', 'Parch', 'pearson'))
file_answer5.write(answer5)
file_answer5.close()

# Question 6
# What is the most popular female name on the ship?
# Extract the passenger’s full name (Name column) from his personal name (First Name)
file_answer6 = open("answer # 6.txt", "w")
answer6 = str(define_most_popular_name('female'))
file_answer6.write(answer6)
file_answer6.close()
