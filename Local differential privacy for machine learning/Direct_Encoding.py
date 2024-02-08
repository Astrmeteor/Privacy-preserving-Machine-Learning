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


def random_response_age_adult(response):
    true_ans = response > 50
    if np.random.randint(0, 2) == 0:
        return true_ans
    else:
        return np.random.randint(0, 2) == 0


def random_response_aggregation_and_estimation(answers):
    false_yeses = len(answers) / 4
    total_yeses = np.sum([1 if r else 0 for r in answers])

    true_yeses = total_yeses - false_yeses

    rr_result = true_yeses * 2
    return rr_result


def playing_with_the_US_census_dataset():
    # URL of the dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # Local file name to save the data
    local_file_name = "../Data/adult.data"

    # Download the dataset if not already downloaded
    download_dataset(data_url, local_file_name)

    # Now load the specified column using numpy
    ages_adult = load_data(local_file_name, usecols=0, delimiter=",")

    total_count = len([i for i in ages_adult])
    age_over_50_count = len([i for i in ages_adult if i > 50])
    print(f"Total count: " + str(total_count))
    print(f"Age over 50 count: " + str(age_over_50_count))
    print(f"Total count - age over 50 count (Younger than or equal to 50 count): " + str(total_count - age_over_50_count))

    # Data perturbation
    perturbed_age_over_50_count = len([i for i in ages_adult if random_response_age_adult(i)])
    print(f"Perturbed age over 50 count: " + str(perturbed_age_over_50_count))
    print(f"Total count - perturbed age over 50 count (Younger than or equal to 50 count): " + str(total_count - perturbed_age_over_50_count))
    print("Relative error in data aggregation and estimation): %.2f%%" %
          ((perturbed_age_over_50_count - age_over_50_count) / age_over_50_count * 100))

    # Data aggregation and estimation
    answers = [True if random_response_age_adult(i) else False for i in ages_adult]
    estimated_age_over_50_count = random_response_aggregation_and_estimation(answers)
    print(f"Perturbed age over 50 count: " + str(int(estimated_age_over_50_count)))
    print(f"Total count - perturbed age over 50 count (Younger than or equal to 50 count): " + str(
        total_count - int(estimated_age_over_50_count)))
    print(f"Relative error in data aggregation and estimation): %.2f%%" %
          ((estimated_age_over_50_count - age_over_50_count) / age_over_50_count * 100))


def number_of_people_in_each_occupation_domain():
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
    adult = pd.read_csv(io.StringIO(adult.decode("utf-8")), header=None, na_values="?", delimiter=r", ", engine="python")
    adult.dropna()
    adult.head()
    domain = adult[6].dropna().unique()
    domain.sort()
    print("Domain:")
    print(domain)

    # the number of people of each occupation
    adult_occupation = adult[6].dropna()
    num_occ = adult_occupation.value_counts().sort_index()
    print("The number of people of each occupation:")
    print(num_occ)

    # Direct encoding
    print("Encoding:")
    print([encoding(domain, i) for i in domain])

    # Perturbation algorithm in direct encoding
    # Use the default epsilon 5.0
    epsilon = 5.0
    print(f"{'Raw':<20}{'Encoding':<10}{'Perturbation':<20}{'Encoding (AP)':<15}{'Identical':<10}{'Epsilon':<10}")
    test_epochs = 10
    idential_ratio = 0
    for i in range(test_epochs):
        ans = random.choice(domain)
        enc = encoding(domain, ans)
        ans_per = encoding_perturbation(domain, enc)
        enc_per = encoding(domain, ans_per)
        ide = ans == ans_per
        idential_ratio += 1 == ide
        print(f"{ans:<20}{enc:<10}{ans_per:<20}{enc_per:<15}{str(ide):<10}{epsilon:<10.2f}")
    print(f"After perturbation with epsilon {epsilon} in {test_epochs} epochs, the identical ratio: {(idential_ratio / test_epochs * 100):.2f}%\n")

    # Change the epsilon to 0.1
    epsilon = 0.1
    idential_ratio = 0
    for i in range(test_epochs):
        ans = random.choice(domain)
        enc = encoding(domain, ans)
        ans_per = encoding_perturbation(domain, enc, epsilon)
        enc_per = encoding(domain, ans_per)
        ide = ans == ans_per
        idential_ratio += 1 == ide
        print(f"{ans:<20}{enc:<10}{ans_per:<20}{enc_per:<15}{str(ide):<10}{epsilon:<10.2f}")
    print(
        f"After perturbation with epsilon {epsilon} in {test_epochs} epochs, the identical ratio: {(idential_ratio / test_epochs * 100):.2f}%")

    # After applying perturbation for the direct encoding
    perturbed_ans = pd.DataFrame([encoding_perturbation(domain, encoding(domain, i)) for i in adult_occupation])
    num_occ_per = perturbed_ans.value_counts().sort_index()
    print("\nThe number of people of each occupation with perturbation after direct encoding:")
    print(num_occ_per)

    num_occ_per.index = num_occ_per.index.get_level_values(0)
    difference = num_occ - num_occ_per
    print("\nAfter perturbation, the difference between perturbed and actual numbers")
    print(difference)

    print("\nApplying aggregation and estimation to direct encoding")
    estimated_answers = direct_encoding_aggregation_and_estimation(domain, perturbed_ans)
    print(list(zip(domain, estimated_answers)))


def encoding(domain, answer):
    return int(np.where(domain == answer)[0])


def encoding_perturbation(domain, encoded_ans, epsilon=5.0):
    d = len(domain)
    p = pow(math.e, epsilon) / (d - 1 + pow(math.e, epsilon))
    q = (1.0 - p) / (d - 1.0)

    s1 = np.random.random()
    if s1 <= p:
        return domain[encoded_ans]
    else:
        s2 = np.random.randint(0, d - 1)
        return domain[(encoded_ans + s2) % d]


def direct_encoding_aggregation_and_estimation(domain, answers, epsilon=5.0):
    n = len(answers)
    d = len(domain)
    p = pow(math.e, epsilon) / (d - 1 + pow(math.e, epsilon))
    q = (1.0 - p) / (d - 1.0)

    aggregator = answers.value_counts().sort_index()

    return [max(int((i - n * q) / (p - q)), 1) for i in aggregator]


if __name__ == "__main__":
    # Randomized response for local differential privacy
    # playing_with_the_US_census_dataset()

    # Direct encoding
    number_of_people_in_each_occupation_domain()