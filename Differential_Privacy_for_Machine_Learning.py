import numpy as np
import diffprivlib
import urllib.request
import os
import matplotlib.pyplot as plt

# Function to download and save the dataset, if it doesn't exist
def download_dataset(url, filename):
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

# Function to load the dataset into a numpy array
def load_data(filename, usecols, delimiter):
    return np.loadtxt(filename, usecols=usecols, delimiter=delimiter)

def counting_query():
    # URL of the dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # Local file name to save the data
    local_file_name = "adult.data"

    # Download the dataset if not already downloaded
    download_dataset(data_url, local_file_name)

    # Now load the specified column using numpy
    ages_adult = load_data(local_file_name, usecols=0, delimiter=",")

    count = len([age for age in ages_adult if age > 50])
    print(f"1. Calculating the number of individuals in the dataset who are more than 50 years old:" + str(count))
    
    sensitivity = 1
    epsilon = 0.1
    # We add noise draw from Lap(sensitivity/epsilon)
    count = len([i for i in ages_adult if i > 50]) + np.random.laplace(loc=0, scale=sensitivity/epsilon)
    print(f"2. Calculating the number of individuals in the dataset who are more than 50 years old with DP:" + str(count))

    sensitivity = 1
    epsilon = 0.001
    # try it with much smaller epsilon
    count = len([i for i in ages_adult if i > 50]) + np.random.laplace(loc=0, scale=sensitivity / epsilon)
    print(f"3. Calculating the number of individuals in the dataset who are more than 50 years old with DP (much smaller epsilon):" + str(count))


def histogram_laplace(sample, epsilon=1, bins=10, range=None, normed=None, weights=None, density=None):
    hist, bin_edges = np.histogram(sample, bins=bins, range=range, weights=weights, density=None)
    dp_mech = diffprivlib.mechanisms.Laplace(epsilon=epsilon, sensitivity=1)
    dp_hist = np.zeros_like(hist)

    for i in np.arange(dp_hist.shape[0]):
        dp_hist[i] = dp_mech.randomise(int(hist[i]))

    if normed or density:
        bin_sizes = np.array(np.diff(bin_edges), float)
        return dp_hist / bin_sizes / dp_hist.sum(), bin_edges

    return dp_hist, bin_edges


def histogram_query():
    local_file_name = "adult.data"

    ages_adult = load_data(local_file_name, usecols=0, delimiter=",")

    hist, bins = np.histogram(ages_adult)
    hist = hist / hist.sum()

    plt.bar(bins[:-1], hist, width=(bins[1] - bins[0]) * 0.9)
    plt.savefig("histogram_query.png")
    # plt.show()

    plt.clf()

    # differentially private histogram query works as epsilon = 0.01
    dp_hist, dp_bins = histogram_laplace(ages_adult, epsilon=0.01)
    dp_hist = dp_hist / dp_hist.sum()

    plt.bar(dp_bins[:-1], dp_hist, width=(dp_bins[1] - dp_bins[0]) * 0.9)
    plt.savefig("histogram_query_dp.png")
    plt.show()


if __name__ == "__main__":
    # Laplac mechanism
    # counting_query()
    histogram_query()

