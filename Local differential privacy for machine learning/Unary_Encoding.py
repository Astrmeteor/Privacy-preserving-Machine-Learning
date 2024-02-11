import io
import math
import random
import numpy as np
import os
import urllib.request
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


def retrieving_different_race_categories():
    # URL of the dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # Local file name to save the data
    local_file_name = "../Data/adult.data"

    # Download the dataset if not already downloaded
    download_dataset(data_url, local_file_name)

    # 首先读取二进制文件的内容
    with open(local_file_name, 'rb') as file:
        adult = file.read()

    # Now load the specified column using numpy
    adult = pd.read_csv(io.StringIO(adult.decode("utf-8")), header=None, na_values="?", delimiter=r", ",
                        engine="python")
    adult.dropna()
    adult.head()
    domain = adult[8].dropna().unique()
    domain.sort()
    print("Domain:")
    print(domain)

    adult_race = adult[8].dropna()
    print(f"\nNumber of the race:\n" + str(adult_race.value_counts().sort_index()))

    # Single test
    print("Test the encoding:")
    print([encoding(domain, i) for i in adult_race[:5]])

    # Single test perturbation
    print("Test the encoding with perturbation in SUE (Epislon 5.0):")
    print([sym_perturbation(encoding(domain, i)) for i in adult_race[:5]])
    print("Test the encoding with perturbation in OUE (Epsilon 5.0):")
    print([sym_perturbation(encoding(domain, i), mode="oue") for i in adult_race[:5]])

    print("Test the encoding with perturbation in SUE (Epislon 0.1):")
    print([sym_perturbation(encoding(domain, i), epsilon=0.1) for i in adult_race[:5]])
    print("Test the encoding with perturbation in OUE (Epsilon 0.1):")
    print([sym_perturbation(encoding(domain, i), epsilon=0.1, mode="oue") for i in adult_race[:5]])

    # Test the perturbed answers
    sym_perturbed_answers = np.sum([sym_perturbation(encoding(domain, r)) for r in adult_race], axis=0)
    print(f"Test the perturbed answers in SUE:\n" + str(list(zip(domain, sym_perturbed_answers))))
    sym_perturbed_answers = np.sum([sym_perturbation(encoding(domain, r), mode="oue") for r in adult_race], axis=0)
    print(f"Test the perturbed answers in OUE:\n" + str(list(zip(domain, sym_perturbed_answers))))

    # Aggregation and estimation
    sym_perturbed_answers = [sym_perturbation(encoding(domain, r)) for r in adult_race]
    estimated_answers = sym_aggregation_and_estimation(sym_perturbed_answers)
    print(f"Data aggregation and estimation in SUE:\n" + str(list(zip(domain, estimated_answers))))

    sym_perturbed_answers = [sym_perturbation(encoding(domain, r), mode="oue") for r in adult_race]
    estimated_answers = sym_aggregation_and_estimation(sym_perturbed_answers, mode="oue")
    print(f"Data aggregation and estimation in SUE:\n" + str(list(zip(domain, estimated_answers))))


def encoding(domain, answer):
    return [1 if d == answer else 0 for d in domain]


def sym_perturbation(encoded_ans, epsilon=5.0, mode="sue"):
    return [sym_perturb_bit(b, epsilon, mode) for b in encoded_ans]


def sym_perturb_bit(bit, epsilon=5.0, mode="sue"):
    # Symmetric unary encoding
    if mode == "sue":
        p = pow(math.e, epsilon / 2) / (1 + pow(math.e, epsilon / 2))
        q = 1 - p

    # Optimal unary encoding
    elif mode == "oue":
        p = 1 /2
        q = 1 / (pow(math.e, epsilon) + 1)

    s = np.random.random()

    if bit == 1:
        if s <= p:
            return 1
        else:
            return 0
    elif bit == 0:
        if s <= q:
            return 1
        else:
            return 0


def sym_aggregation_and_estimation(answers, epsilon=5.0, mode="sue"):
    # Symmetric unary encoding
    if mode == "sue":
        p = pow(math.e, epsilon / 2) / (1 + pow(math.e, epsilon / 2))
        q = 1 - p

    # Optimal unary encoding
    elif mode == "oue":
        p = 1 / 2
        q = 1 / (pow(math.e, epsilon) + 1)

    sums = np.sum(answers, axis=0)
    n = len(answers)

    return [int((i - n * q) / (p - q)) for i in sums]


if __name__ == "__main__":
    # Unary Encoding
    retrieving_different_race_categories()