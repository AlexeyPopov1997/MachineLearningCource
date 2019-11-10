import pandas
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


# Function for filling the classification error table
def fill_clf_error_table(data: pandas.DataFrame):
    true_positive = data[(data['pred'] == 1) & (data['true'] == 1)]
    false_positive = data[(data['pred'] == 1) & (data['true'] == 0)]
    false_negative = data[(data['pred'] == 0) & (data['true'] == 1)]
    true_negative = data[(data['pred'] == 0) & (data['true'] == 0)]
    return len(true_positive), len(false_positive), len(false_negative), len(true_negative)


# Function for calculating the basic quality metrics of the classifier
def calc_basic_metric(data: pandas.DataFrame):
    accuracy = accuracy_score(data['true'], data['pred'])
    precision = precision_score(data['true'], data['pred'])
    recall = recall_score(data['true'], data['pred'])
    f_score = f1_score(data['true'], data['pred'])
    return accuracy, precision, recall, f_score


# The calculation of the area under the ROC curve for each classifier
def calc_auc_roc(scores: pandas.DataFrame):
    clf_names = scores.columns[1:]
    scores = pandas.Series([roc_auc_score(scores['true'], scores[clf]) for clf in clf_names], index=clf_names)
    clf_name = scores.sort_values(ascending=False).index[0]
    return clf_names, scores, clf_name


# Function for determining the classifier with the highest precision with recall of at least 70%
def define_max_recall_clf(clf_names, scores):
    pr = []
    for clf in clf_names:
        if scores[clf] <= 0.7:
            pr.append([scores[clf], clf])
    return max(pr)[1]


def main():
    data = pandas.read_csv('classification.csv')
    true_positive, false_positive, false_negative, true_negative = fill_clf_error_table(data)
    print('Classification error table')
    print('True Positive:', true_positive)
    print('False Positive:', false_positive)
    print('False Negative:', false_negative)
    print('True Negative:', true_negative, '\n')
    write_answer('1.txt', str(true_positive) + ' ' + str(false_positive) +
                 ' ' + str(false_negative) + ' ' + str(true_negative))

    accuracy, precision, recall, f_score = calc_basic_metric(data)
    print('Basic classifier accuracy metrics')
    print('Accuracy:', accuracy.round(2))
    print('Precision:', precision.round(2))
    print('Recall:', recall.round(2))
    print('F-score:', f_score.round(2), '\n')
    write_answer('2.txt', str(accuracy.round(2)) + ' ' + str(precision.round(2)) +
                 ' ' + str(recall.round(2)) + ' ' + str(f_score.round(2)))

    scores = pandas.read_csv('scores.csv')
    clf_names, scores, clf_name = calc_auc_roc(scores)
    clf_70_percent = define_max_recall_clf(clf_names, scores)
    print('Classifier with the highest AUC-ROC metric value:', clf_name)
    write_answer('3.txt', str(clf_name))
    print('Classifiers with the highest precision with recall of at least 70%:', clf_70_percent)
    write_answer('4.txt', str(clf_70_percent))


if __name__ == '__main__':
    main()
