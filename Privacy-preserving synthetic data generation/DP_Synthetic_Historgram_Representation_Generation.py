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
import ast

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


def dp_synthetic_histogram_representation_generation():
    # URL of the dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    req = requests.get(data_url).content
    # Local file name to save the data
    local_file_name = "../Data/adult.data"

    # Download the dataset if not already downloaded
    download_dataset(data_url, local_file_name)

    # Now load the specified column using numpy
    # 首先读取二进制文件的内容
    with open(local_file_name, 'rb') as file:
        adult = file.read()

    # Now load the specified column using numpy
    adult = pd.read_csv(io.StringIO(adult.decode("utf-8")), header=None, na_values="?", delimiter=r", ", engine="python")
    adult.dropna()
    adult.head()
    # print data
    # print(adult)
    adult_age = adult[0].dropna()
    # adult_age = adult[0].apply(lambda x: x[1] if isinstance(x, tuple) else x)
    adult_age = [adult_age[i] for i in range(len(adult_age))]
    print("Age count between 44 and 55: " + str(age_count_query(adult_age, lo=44, hi=55)))

    # Domain of ages
    age_domain = list(range(0, 100))
    age_histogram = [age_count_query(adult_age, age, age + 1) for age in age_domain]
    plt.bar(age_domain, age_histogram)
    plt.ylabel("The number of people (Frequency)")
    plt.xlabel("Ages")
    # plt.show()
    plt.savefig("number_of_people_frequency.png")
    plt.clf()

    # Implementing a count query using a synthetic histogram generator
    print("A count query using a synthetic histogram generator:" + str(synthetic_age_count_query(age_histogram, lo=44, hi=55)))

    # Adding the Laplace mechanism
    sensitivity = 1.0
    epsilon = 1.0
    dp_age_histogram = [laplace_mechnism(age_count_query(adult_age, age, age + 1), sensitivity, epsilon) for age in age_domain]
    plt.bar(age_domain, dp_age_histogram)
    plt.ylabel("The number of people (Frequency)")
    plt.xlabel("Ages")
    plt.show()
    plt.savefig("differentially_private_synthetic_histogram.png")

    print("Generate a differentially private count query result using the differentially synthetic histogram: " + str(synthetic_age_count_query(dp_age_histogram, lo=44, hi=55)))


def age_count_query(adult_age, lo, hi):
    return sum(1 for age in adult_age if lo <= age < hi)


def synthetic_age_count_query(syn_age_hist_rep, lo, hi):
    return sum(syn_age_hist_rep[age] for age in range(lo, hi))


def laplace_mechnism(data, sensitivity, epsilon):
    return data + np.random.laplace(loc=0, scale=sensitivity / epsilon)


if __name__ == "__main__":
    dp_synthetic_histogram_representation_generation()