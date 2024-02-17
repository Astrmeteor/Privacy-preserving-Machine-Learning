import numpy as np
import diffprivlib
import urllib.request
import os
import pandas as pd
import matplotlib.pyplot as plt
import os

# 获取当前脚本文件的完整路径
script_path = os.path.abspath(__file__)

# 获取脚本所在目录的路径
script_dir = os.path.dirname(script_path)

# 将当前工作目录更改为脚本所在目录
os.chdir(script_dir)


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
    local_file_name = "../Data/adult.data"

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


def number_of_people():
    adult = pd.read_csv("../Data/adult.csv")

    print("Married-civ-spouse: " + str(len([i for i in adult["marital-status"] if i == "Married-civ-spouse"])))
    print("Never-married: " + str(len([i for i in adult["marital-status"] if i == "Never-married"])))
    print("Divorced: " + str(len([i for i in adult["marital-status"] if i == "Divorced"])))
    print("Separated: " + str(len([i for i in adult["marital-status"] if i == "Separated"])))
    print("Widowed: " + str(len([i for i in adult["marital-status"] if i == "Widowed"])))
    print("Married-spouse-absent: " + str(len([i for i in adult["marital-status"] if i == "Married-spouse-absent"])))
    print("Married-AF-spouse: " + str(len([i for i in adult["marital-status"] if i == "Married-AF-spouse"])))

    sets = adult["marital-status"].unique()
    res_1 = most_common_marital_exponential(adult['marital-status'], sets, utility, 1, 1)
    print(f"One time query for most common marital status: " + str(res_1))

    res_10000 = [most_common_marital_exponential(adult['marital-status'], sets, utility, 1, 0.1) for i in range(10000)]
    res_10000 = pd.Series(res_10000).value_counts()
    print(f"10,000 times query for most common marital status:\n" + str(res_10000))


def utility(data, sets):
    return data.value_counts()[sets] / 1000


# H(x, a) -> utility function
# x is one of the seven marital categories, and a is the most common marital status
def most_common_marital_exponential(x, A, H, sensitivity, epsilon):
    # calculate the utility for each element of A
    utilities = [H(x, a) for a in A]

    # calculate the probability for each element based on its utility
    probabilities = [np.exp(epsilon * utility / (2 * sensitivity)) for utility in utilities]

    # normalize the probabilities so they sum to 1
    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

    # choose an element from A based on the probabilities
    return np.random.choice(A, 1, p=probabilities)[0]


def utility_vote(data, sets):
    return data.get(sets)


def votes_exponential(x, A, H, sensitivity, epsilon):
    utilities = [H(x, a) for a in A]

    probabilities = [np.exp(epsilon * utility / (2 * sensitivity)) for utility in utilities]

    probabilities = probabilities / np.linalg.norm(probabilities, ord=1)

    return np.random.choice(A, 1, p=probabilities)[0]


def differential_private_vote_query():
    X = {"Football": 49, "Volleybal": 25, "Basketball": 6, "Swimming": 2}
    A = list(X.keys())
    res_1 = votes_exponential(X, A, utility_vote, 1, 1)

    print(f"One time query for vote query: " + str(res_1))

    res_10000 = [votes_exponential(X, A, utility_vote, 1, 1) for i in range(10000)]
    res_10000 = pd.Series(res_10000).value_counts()
    print(f"10,000 times query for vote query:\n" + str(res_10000))


def dp_F(x, sensitivity=1, epsilon=1.0):
    return x + np.random.laplace(loc=0, scale=sensitivity / epsilon)


def applying_sequential_composition():
    local_file_name = "adult.data"

    ages_adult = load_data(local_file_name, usecols=0, delimiter=",")

    sensitivity = 1
    epsilon_1 = 0.1
    epsilon_2 = 0.2
    epsilon_3 = epsilon_1 + epsilon_2

    x = len([i for i in ages_adult if i > 50])

    # plot F1
    plt.hist([dp_F(x, sensitivity=sensitivity, epsilon=epsilon_1) for i in range(1000)], bins=50, label="F1")
    # plot F2
    plt.hist([dp_F(x, sensitivity=sensitivity, epsilon=epsilon_2) for i in range(1000)], bins=50, alpha=.7, label="F2")
    # plot F3
    plt.hist([dp_F(x, sensitivity=sensitivity, epsilon=epsilon_3) for i in range(1000)], bins=50, alpha=.7, label="F3")
    plt.legend()
    plt.savefig("three_dp_function_sequential_composition.png")
    # plt.show()

    plt.clf()

    # plot F_seq
    plt.hist([(dp_F(x, sensitivity=sensitivity, epsilon=epsilon_1) + dp_F(x, sensitivity=sensitivity, epsilon=epsilon_2))/2
              for i in range(1000)], bins=50, alpha=.7, label="F_seq")
    # plot F3
    plt.hist([dp_F(x, sensitivity=sensitivity, epsilon=epsilon_3) for i in range(1000)], bins=50, alpha=.7, label="F3")
    plt.legend()
    plt.savefig("sequential_composition.png")
    plt.show()

    plt.close()


if __name__ == "__main__":
    # Laplac mechanism
    counting_query()
    # histogram_query()

    # Exponential mechanism
    # number_of_people()

    # Differential private votes query
    # differential_private_vote_query()

    # Applying sequential composition
    applying_sequential_composition()
