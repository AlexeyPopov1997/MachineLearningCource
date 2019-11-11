import pandas
from sklearn.decomposition import PCA
import numpy


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


# Function for determining the first component of a PCA conversion
def get_first_component(pca):
    global index
    sum_var = 0
    for index, var in enumerate(pca.explained_variance_ratio_):
        sum_var += var
        if sum_var >= 0.9:
            break
    first_component_name = str(index + 1)
    return first_component_name


def main():
    close_prices = pandas.read_csv("close_prices.csv")
    close_prices.head()
    x = close_prices.loc[:, 'AXP':]

    # PCA conversion training with a component count of 10
    pca = PCA(n_components=10)
    pca.fit(x)

    # Application of the constructed transformation to the source data
    x_0 = pandas.DataFrame(pca.transform(x))[0]
    x_0.head()

    dji_index = pandas.read_csv("djia_index.csv")
    dji_index.head()

    # Pearson correlation between the first component and the Dow Jones index
    corr = numpy.corrcoef(x_0, dji_index['^DJI'])

    write_answer('1.txt', str(get_first_component(pca)))
    print('Number of components to describe 90% dispersion:', str(get_first_component(pca)))

    write_answer('2.txt', str(corr[1, 0].round(2)))
    print('The pearson correlation between the first component and the Dow Jones index:', str(corr[1, 0].round(2)))

    write_answer('3.txt', str(x.columns[numpy.argmax(pca.components_[0])]))
    print('The company with the highest weight in the first component:', str(x.columns[numpy.argmax(pca.components_[0])]))


if __name__ == '__main__':
    main()
