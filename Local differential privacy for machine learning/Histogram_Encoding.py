import sys
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import os
import io
import urllib.request
import requests
import pandas as pd

# 获取当前脚本文件的完整路径
import requests

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


def histogram_encoding():
    # URL of the dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # Local file name to save the data
    local_file_name = "../Data/adult.data"

    # Download the dataset if not already downloaded
    download_dataset(data_url, local_file_name)

    # Now load the specified column using numpy
    # 首先读取二进制文件的内容
    with open(local_file_name, 'rb') as file:
        adult = file.read()

    # Now load the specified column using numpy
    adult = pd.read_csv(io.StringIO(adult.decode("utf-8")), header=None, na_values="?", delimiter=r", ",
                        engine="python")
    adult.dropna()
    adult.head()

    adult_age = adult[0].dropna()
    ax = adult_age.plot.hist(bins=100, alpha=1.0)
    # plt.show()
    plt.savefig("histogram_encoding.png")
    plt.clf()

    # Define the input domain, i.e., the survey answers
    domain = np.arange(10, 101)
    domain.sort()
    print(f"Domain: " + str(domain))

    # Single value test
    # print(encoding(domain, 11))

    # Summation with histogram encoding, single value test
    print(f"Summation with histogram encoding (epsilon = 5.0): " + str(she_perturbation(encoding(domain, 11))))
    print(f"Summation with histogram encoding (epsilon = 1.0): " + str(she_perturbation(encoding(domain, 11), epsilon=1.0)))

    # Summation with histogram encoding, estimation
    she_estimated_answers = np.sum([she_perturbation(encoding(domain, r)) for r in adult_age], axis=0)
    plt.bar(domain, she_estimated_answers)
    # plt.show()
    plt.savefig("summation_with_histogram_encoding.png")
    plt.clf()

    # Thresholding with histogram encoding, single value test
    print(f"Thresholding with histogram encoding (epsilon = 5.0): " + str(the_perturbation(encoding(domain, 11))))
    print(f"Thresholding with histogram encoding (epsilon = 1.0): " + str(the_perturbation(encoding(domain, 11), epsilon=1.0)))

    # Thresholding with histogram encoding, estimation
    the_perturbed_answers = np.sum([the_perturbation(encoding(domain, r)) for r in adult_age], axis=0)
    plt.bar(domain, the_perturbed_answers)
    plt.ylabel("Frequency")
    plt.xlabel("Ages")
    plt.show()
    plt.savefig("thresholding_with_histogram_encoding.png")
    plt.clf()

    # Data aggregation and estimation
    the_perturbed_answers = [the_perturbation(encoding(domain, r)) for r in adult_age]
    estimated_answers = the_aggregation_and_estimation(the_perturbed_answers)
    plt.bar(domain, estimated_answers)
    plt.ylabel("Frequency")
    plt.xlabel("Ages")
    plt.show()
    plt.savefig("aggregation_and_estimation.png")
    plt.clf()


def encoding(domain, answer):
    return [1.0 if d == answer else 0.0 for d in domain]


# Summation with histogram encoding
def she_perturbation(encoded_ans, epsilon=5.0):
    return [she_perturb_bit(b, epsilon) for b in encoded_ans]


def she_perturb_bit(bit, epsilon=5.0):
    return bit + np.random.laplace(loc=0, scale=2 / epsilon)


# Thresholding with histogram encoding
def the_perturbation(encoded_ans, epsilon=5.0, theta=1.0):
    return [the_perturb_bit(b, epsilon, theta) for b in encoded_ans]


def the_perturb_bit(bit, epsilon=5.0, theta=1.0):
    val = bit + np.random.laplace(loc=0, scale=2 / epsilon)

    if val > theta:
        return 1.0
    else:
        return 0.0


def the_aggregation_and_estimation(answers, epsilon=5.0, theta=1.0):
    p = 1 - 0.5 * pow(math.e, epsilon / 2 * (1.0 - theta))
    q = 0.5 * pow(math.e, epsilon / 2 * (0.0 - theta))

    sums = np.sum(answers, axis=0)
    n = len(answers)

    return [int((i - n * q) / (p - q)) for i in sums]


if __name__ == "__main__":
    # Historgram encoding
    histogram_encoding()
