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
    # adult_age = adult[0].apply(lambda x: 1] if isinstance(x, tuple) else x)
    adult_age = [adult_age[i] for i in range(len(adult_age))]
    print("Age count between 44 and 55: " + str(age_count_query(adult_age, lo=44, hi=55)))

    # Domain of ages
    age_domain = list(range(0, 100))
    age_histogram = [age_count_query(adult_age, age, age + 1) for age in age_domain]
    plt.bar(age_domain, age_histogram)
    plt.ylabel("The number of people (Frequency)")
    plt.xlabel("Ages")
    plt.savefig("number_of_people_frequency.png")
    # plt.show()
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
    plt.savefig("differentially_private_synthetic_histogram.png")
    # plt.show()
    plt.clf()

    print("Generate a differentially private count query result using the differentially synthetic histogram: " + str(synthetic_age_count_query(dp_age_histogram, lo=44, hi=55)))

    # Preprocessing and normalization operation
    dp_age_histogram_preprocessed = np.clip(dp_age_histogram, 0, None)
    dp_age_histogram_normalized = dp_age_histogram_preprocessed / np.sum(dp_age_histogram_preprocessed)
    plt.bar(age_domain, dp_age_histogram_normalized)
    plt.ylabel("Frequency Rates (probabilities)")
    plt.xlabel("Ages")
    plt.savefig("the_probability_histogram.png")
    # plt.show()
    plt.clf()

    # Generating the differentially private synthetic tabular data
    syn_tabular_data = pd.DataFrame(syn_tabular_data_gen(age_domain, 10, dp_age_histogram_normalized), columns=["Age"])
    print("Differentially private synthetic tabular data:\n" + str(syn_tabular_data))

    # The histogram of the synthetic data
    syn_tabular_data = pd.DataFrame(syn_tabular_data_gen(age_domain, len(adult_age), dp_age_histogram_normalized), columns=["Age"])
    plt.hist(syn_tabular_data["Age"], bins=age_domain)
    plt.ylabel("The number of people (Frequency) - Synthetic")
    plt.xlabel("Ages")
    plt.savefig("the_histogram_of_the_synthetic_data.png")
    # plt.show()
    plt.clf()

    # The histogram of the original data
    plt.bar(age_domain, age_histogram)
    plt.ylabel("The number of people (Frequency) - True Value")
    plt.xlabel("Ages")
    plt.savefig("the_histogram_of_the_true_data.png")
    # plt.show()
    plt.clf()

    # The 2-way marginal representation
    two_way_marginal_rep = adult.groupby([0, 5]).size().reset_index(name="count")
    print("The 2-way marginal representation")
    print(two_way_marginal_rep)

    dp_two_way_marginal_rep = laplace_mechnism(two_way_marginal_rep["count"], 1, 1)
    print("The differentially privat 2-way marginal representation")
    print(dp_two_way_marginal_rep)

    # Generating synthetic multi-marginal data
    dp_two_way_marginal_rep_preprocessed = np.clip(dp_two_way_marginal_rep, 0, None)
    dp_two_way_marginal_rep_normalized = dp_two_way_marginal_rep_preprocessed / np.sum(dp_two_way_marginal_rep_preprocessed)
    print("Normalized dp two-way marginal representation")
    print(dp_two_way_marginal_rep_normalized)

    age_marital_pairs = [(a, b) for a,b,_ in two_way_marginal_rep.values.tolist()]
    list(zip(age_marital_pairs, dp_two_way_marginal_rep_normalized))

    set_of_potential_samples = range(0, len(age_marital_pairs))

    n = laplace_mechnism(len(adult), 1.0, 1.0)

    generating_synthetic_data_samples = np.random.choice(set_of_potential_samples, int(max(n, 0)), p=dp_two_way_marginal_rep_normalized)
    synthetic_data_set = [age_marital_pairs[i] for i in generating_synthetic_data_samples]

    synthetic_data = pd.DataFrame(synthetic_data_set, columns=["Age", "Marital status"])
    print("Synthetic data")
    print(synthetic_data)

    # The histogram produced using synthetic multi-marginal data
    plt.hist(synthetic_data["Age"], bins=age_domain)
    plt.ylabel("The number of people (Frequency) - Synthetic")
    plt.xlabel("Ages")
    plt.savefig("histogram_produced_using_synthetically_generated_multimarginal_data.png")
    # plt.show()
    plt.clf()

    # The statistics of the original data
    adult_marital_status = adult[5].dropna()
    print("The statistics of the original data")
    print(adult_marital_status.value_counts().sort_index())

    # The statistics of the synthetic data
    syn_adult_marital_status = synthetic_data["Marital status"].dropna()
    print("The statistics of the synthetic data")
    print(syn_adult_marital_status.value_counts().sort_index())

    # Exercise 1
    # The education and occupation columns
    edu_occ = adult.groupby([3, 6]).size().reset_index(name="count")
    print("The 2-way marginal representation of Education and Occupation")
    print(edu_occ)

    dp_edu_occ = laplace_mechnism(edu_occ["count"], 1, 1)
    print("The differentially privat 2-way marginal representation of Education and Occupation")
    print(dp_edu_occ)

    # Generating synthetic multi-marginal data of Edu and Occ
    dp_edu_occ_prepocessed = np.clip(dp_edu_occ, 0, None)
    dp_edu_occ_normalized = dp_edu_occ_prepocessed / np.sum(
        dp_edu_occ_prepocessed)
    print("Normalized dp two-way marginal representation of Edu and Occ")
    print(dp_edu_occ_normalized)

    edu_occ_pairs = [(a, b) for a, b, _ in edu_occ.values.tolist()]
    list(zip(edu_occ_pairs, dp_edu_occ_normalized))

    set_of_potential_samples_edu_occ = range(0, len(edu_occ_pairs))

    n = laplace_mechnism(len(adult), 1.0, 1.0)

    generating_synthetic_data_samples_edu_occ = np.random.choice(set_of_potential_samples_edu_occ, int(max(n, 0)),
                                                         p=dp_edu_occ_normalized)
    synthetic_data_set_edu_occ = [edu_occ_pairs[i] for i in generating_synthetic_data_samples_edu_occ]

    synthetic_data_edu_occ = pd.DataFrame(synthetic_data_set_edu_occ, columns=["Education", "Occupation"])
    print("Synthetic data of Education and Occupation")
    print(synthetic_data_edu_occ)

    # The histogram produced using synthetic multi-marginal data
    edu_domain = [
        'Preschool',
        '1st-4th',
        '5th-6th',
        '7th-8th',
        '9th',
        '10th',
        '11th',
        '12th',  # 注意12th通常认为是高中，但没有毕业
        'HS-grad',  # 高中毕业
        'Some-college',  # 一些大学教育但没有学位
        'Assoc-voc',  # 职业学校学位
        'Assoc-acdm',  # 学院学位
        'Bachelors',  # 学士学位
        'Masters',  # 硕士学位
        'Prof-school',  # 专业学校，如法律和医学
        'Doctorate'  # 博士学位
    ]

    # 这里我们创建一个映射，将教育程度映射为整数
    edu_mapping = {domain: idx for idx, domain in enumerate(edu_domain)}
    # 将教育程度列转换为对应的数值
    education_numeric = synthetic_data_edu_occ["Education"].map(edu_mapping)

    plt.hist(education_numeric, bins=np.arange(len(edu_domain)+1)-0.5)
    plt.ylabel("The number of people (Frequency) - Synthetic")
    plt.xlabel("Education")
    plt.savefig("histogram_produced_using_synthetically_generated_multimarginal_data_edu_occ.png")
    # plt.show()
    plt.clf()

    # The statistics of the original data Edu
    adult_edu_status = adult[6].dropna()
    print("The statistics of the original data")
    print(adult_edu_status.value_counts().sort_index())

    # The statistics of the synthetic data Edu
    syn_adult_edu_status = synthetic_data_edu_occ["Occupation"].dropna()
    print("The statistics of the synthetic data")
    print(syn_adult_edu_status.value_counts().sort_index())

    # Exercise 2, try 3-way marginal representation with 3-column data
    # The 3-way marginal representation, 0 age， 5 marital status， 6 occupation
    three_way_marginal_rep = adult.groupby([0, 5, 6]).size().reset_index(name="count")
    print("The 3-way marginal representation")
    print(three_way_marginal_rep)

    dp_three_way_marginal_rep = laplace_mechnism(three_way_marginal_rep["count"], 1, 1)
    print("The differentially privat 3-way marginal representation")
    print(dp_three_way_marginal_rep)

    # Generating synthetic multi-marginal data
    # compute how probabilities of different classes
    dp_three_way_marginal_rep_preprocessed = np.clip(dp_three_way_marginal_rep, 0, None)
    dp_three_way_marginal_rep_normalized = dp_three_way_marginal_rep_preprocessed / np.sum(
        dp_three_way_marginal_rep_preprocessed)
    print("Normalized dp three-way marginal representation")
    print(dp_three_way_marginal_rep_normalized)

    age_marital_occ_combinations = [(a, b, c) for a, b, c, _ in three_way_marginal_rep.values.tolist()]
    list(zip(age_marital_occ_combinations, dp_three_way_marginal_rep_normalized))

    set_of_potential_samples = range(0, len(age_marital_occ_combinations))

    n = laplace_mechnism(len(adult), 1.0, 1.0)

    generating_synthetic_data_samples = np.random.choice(set_of_potential_samples, int(max(n, 0)),
                                                         p=dp_three_way_marginal_rep_normalized)
    synthetic_data_set = [age_marital_occ_combinations[i] for i in generating_synthetic_data_samples]

    synthetic_data = pd.DataFrame(synthetic_data_set, columns=["Age", "Marital status", "Occupation"])
    print("Synthetic data")
    print(synthetic_data)

    # The histogram produced using synthetic multi-marginal data
    plt.hist(synthetic_data["Age"], bins=age_domain)
    plt.ylabel("The number of people (Frequency) - Synthetic")
    plt.xlabel("Ages")
    plt.savefig("histogram_produced_using_synthetically_generated_multimarginal_data_3way.png")
    # plt.show()
    plt.clf()

    # The statistics of the original data
    adult_marital_status = adult[0].dropna()
    print("The statistics of the original data: Age")
    print(adult_marital_status.value_counts().sort_index())

    # The statistics of the synthetic data
    syn_adult_marital_status = synthetic_data["Age"].dropna()
    print("The statistics of the synthetic data: Age")
    print(syn_adult_marital_status.value_counts().sort_index())

    adult_marital_status = adult[5].dropna()
    print("The statistics of the original data: Marital Status")
    print(adult_marital_status.value_counts().sort_index())

    # The statistics of the synthetic data
    syn_adult_marital_status = synthetic_data["Marital status"].dropna()
    print("The statistics of the synthetic data: Marital status")
    print(syn_adult_marital_status.value_counts().sort_index())

    adult_marital_status = adult[6].dropna()
    print("The statistics of the original data: Occupation")
    print(adult_marital_status.value_counts().sort_index())

    # The statistics of the synthetic data
    syn_adult_marital_status = synthetic_data["Occupation"].dropna()
    print("The statistics of the synthetic data: Occupation")
    print(syn_adult_marital_status.value_counts().sort_index())


def age_count_query(adult_age, lo, hi):
    return sum(1 for age in adult_age if lo <= age < hi)


def synthetic_age_count_query(syn_age_hist_rep, lo, hi):
    return sum(syn_age_hist_rep[age] for age in range(lo, hi))


def laplace_mechnism(data, sensitivity, epsilon):
    return data + np.random.laplace(loc=0, scale=sensitivity / epsilon)


# cnt, the number of choice
def syn_tabular_data_gen(age_domain, cnt, dp_age_histogram_normalized):
    return np.random.choice(age_domain, cnt, p=dp_age_histogram_normalized)


if __name__ == "__main__":
    dp_synthetic_histogram_representation_generation()