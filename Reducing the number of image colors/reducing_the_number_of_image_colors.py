import numpy
import pandas
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage.io import imread, imshow
from sklearn.cluster import KMeans


class Image:
    def __init__(self, image_name):
        image = img_as_float(imread(image_name))
        width, height, depth = image.shape
        pixels = pandas.DataFrame(numpy.reshape(image, (width * height, depth)), columns=['R', 'G', 'B'])
        self.image = image
        self.width = width
        self.height = height
        self.depth = depth
        self.pixels = pixels


# Function to write an answer to a text file
def write_answer(file_name, answer):
    file_answer = open(file_name, 'w')
    file_answer.write(answer)
    file_answer.close()


# Function to run the K-Means algorithm
def cluster_pixels(pixels: pandas.DataFrame, n_clusters: int = 8) -> pandas.DataFrame:
    pixels = pixels.copy()
    model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=241)
    pixels['cluster'] = model.fit_predict(pixels)
    return pixels


# Function for filling pixels in two ways: mean color across the cluster and median
def mean_median_image(pixels: pandas.DataFrame, width, height, depth):
    means = pixels.groupby('cluster').mean().values
    mean_pixels = numpy.array([means[c] for c in pixels['cluster']])
    mean_image = numpy.reshape(mean_pixels, (width, height, depth))
    medians = pixels.groupby('cluster').median().values
    median_pixels = numpy.array([medians[c] for c in pixels['cluster']])
    median_image = numpy.reshape(median_pixels, (width, height, depth))
    return mean_image, median_image


# Function for measuring the quality of the resulting segmentation using the PSNR metric
def define_PSNR(image1: numpy.array, image2: numpy.array):
    mse = numpy.mean((image1 - image2) ** 2)
    return 10.0 * numpy.log10(1.0 / mse)


def show_images(mean_image: numpy.array, median_image: numpy.array):
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.title.set_text('Mean image')
    imshow(mean_image)
    ax = fig.add_subplot(1, 2, 2)
    ax.title.set_text('Median image')
    imshow(median_image)
    plt.show()


# Function for finding the minimum number of clusters at which the PSNR value is above 20
def define_min_num_clusters(image, width, height, depth, pixels: pandas.DataFrame, n=None):
    for n in range(1, 21):
        clustersPixels = cluster_pixels(pixels, n)
        mean_image, median_image = mean_median_image(clustersPixels, width, height, depth)
        show_images(mean_image, median_image)
        psnr_mean, psnr_median = define_PSNR(image, mean_image), define_PSNR(image, median_image)
        if psnr_mean > 20 or psnr_median > 20:
            print('Minimum number of clusters at which the PSNR value is above 20:', str(n))
            break
    return n


def main():
    image = Image('parrots.jpg')
    min_num_clusters = define_min_num_clusters(image.image, image.width, image.height, image.depth, image.pixels)
    write_answer('1.txt', str(min_num_clusters))


if __name__ == '__main__':
    main()
