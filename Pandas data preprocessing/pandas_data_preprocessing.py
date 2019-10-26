import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')


# Display data frame
def show_data():
    print(data)


# Determination of the number of men on the ship
def males_count_define():
    malesCount = data[data.Sex == 'male']['Sex'].count()
    print('Males count: ', malesCount)
    return malesCount


# Determination of the number of women on the ship
def females_count_define():
    femalesCount = data[data.Sex == 'female']['Sex'].count()
    print('Females count: ', femalesCount)
    return femalesCount


# Determination of the survived
# If the value of roundValue = None, displays the number of survivors,
# if not, then the percentage of survivors is displayed, rounded in accordance with the value of roundValue
def survived_define(roundValue=None):
    if roundValue is not None:
        survivedCount = data[data.Survived == 1]['Survived'].count()
        totalPassengersCount = data['Survived'].count()
        survivedCountPercent = 100 * survivedCount / totalPassengersCount
        print('Count of survived (percent): ', survivedCountPercent.round(roundValue))
        return survivedCountPercent.round(roundValue)
    else:
        survivedCount = data[data.Survived == 1]['Survived'].count()
        print('Count of survived: ', survivedCount)
        return survivedCount


# Passenger class definition
# If the value roundValue = None,
# the number of passengers of a certain class (specified using the classValue parameter) is displayed,
# if not, the percentage of passengers is displayed, a specific class, rounded according to the value of roundValue
def class_passengers_define(classValue, roundValue=None):
    if roundValue is not None:
        firstClassPassengersCount = data[data.Pclass == classValue]['Pclass'].count()
        totalPassengersCount = data['Pclass'].count()
        firstClassPassengersCountPercent = 100 * firstClassPassengersCount / totalPassengersCount
        print(classValue, 'st class passengers (percent): ', firstClassPassengersCountPercent.round(roundValue))
        return firstClassPassengersCountPercent.round(roundValue)
    else:
        firstClassPassengersCount = data[data.Pclass == classValue]['Pclass'].count()
        print(classValue, 'st class passengers: ', firstClassPassengersCount)
        return firstClassPassengersCount


# Determining the average age of passengers
def average_age_define(roundValue=None):
    if roundValue is not None:
        averageAge = round(data['Age'].mean(), roundValue)
        print('Average age of passengers:', averageAge)
        return averageAge
    else:
        averageAge = round(data['Age'].mean(), 2)
        print('Average age of passengers:', averageAge)
        return averageAge


# Determining the median age of passengers
def median_age_define(roundValue=None):
    if roundValue is not None:
        medianAge = round(data['Age'].median(), roundValue)
        print('Median age of passengers:', medianAge)
        return medianAge
    else:
        medianAge = round(data['Age'].median(), 2)
        print('Median age of passengers:', medianAge)
        return medianAge


# Counting the correlation value between two columns
def correlation_calculate(firstColumn, secondColumn, method, roundValue=None):
    if roundValue is not None:
        corrValue = round(data[firstColumn].corr(data[secondColumn], method=method), roundValue)
        print('Pearson correlation value:', corrValue)
        return corrValue
    else:
        corrValue = round(data[firstColumn].corr(data[secondColumn], method=method), 2)
        print('Pearson correlation value:', round(data[firstColumn].corr(data[secondColumn], method=method), 2))
        return corrValue


# Identify the most popular name
def most_popular_name_define(sex):
    femaleNames = data[data.Sex == sex]['Name']
    temp = []
    for index in femaleNames:
        if '(' in index:
            if ')' in index.split('(')[1].split(' ')[0]:
                temp.append(index.split('(')[1].split(' ')[0].split(')')[0])
            else:
                temp.append(index.split('(')[1].split(' ')[0])
        else:
            temp.append(index.split('. ')[1].split(' ')[0])
    mostPopularName = str(pandas.DataFrame.from_dict(temp)[0].value_counts()[:1])
    index = 0
    while mostPopularName[index] != ' ':
        index = index + 1
    print('Most popular name:', mostPopularName[:index])
    return mostPopularName[:index]


# Answers on questions

# Question 1
# How many men and women rode the ship? As an answer, give two numbers separated by a space
file_answer1 = open("answer # 1.txt", "w")
answer1 = str(males_count_define()) + ' ' + str(females_count_define())
file_answer1.write(answer1)
file_answer1.close()

# Question 2
# How many passengers survived?
# Calculate the proportion of surviving passengers.
# Give the answer in percent (a number in the range from 0 to 100, a percent sign is not needed),
# rounded to two characters
file_answer2 = open("answer # 2.txt", "w")
answer2 = str(survived_define(2))
file_answer2.write(answer2)
file_answer2.close()

# Question 3
# What is the proportion of first-class passengers among all passengers?
# Give the answer in percent (a number in the range from 0 to 100, a percent sign is not needed),
# rounded to two characters
file_answer3 = open("answer # 3.txt", "w")
answer3 = str(class_passengers_define(1, 2))
file_answer3.write(answer3)
file_answer3.close()

# Question 4
# How old were the passengers?
# Calculate the average and median age of passengers.
# As an answer, give two numbers separated by a space
file_answer4 = open("answer # 4.txt", "w")
answer4 = str(average_age_define(2)) + ' ' + str(median_age_define(2))
file_answer4.write(answer4)
file_answer4.close()

# Question 5
# Do the number of brothers / sisters / spouses correlate with the number of parents / children?
# Count the Pearson correlation between the SibSp and Parch traits
file_answer5 = open("answer # 5.txt", "w")
answer5 = str(correlation_calculate('SibSp', 'Parch', 'pearson'))
file_answer5.write(answer5)
file_answer5.close()

# Question 6
# What is the most popular female name on the ship?
# Extract the passenger’s full name (Name column) from his personal name (First Name)
file_answer6 = open("answer # 6.txt", "w")
answer6 = str(most_popular_name_define('female'))
file_answer6.write(answer6)
file_answer6.close()
