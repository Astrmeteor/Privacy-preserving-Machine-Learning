import numpy as np
import diffprivlib
import urllib.request
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

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
def load_data(filename, usecols, delimiter, dtype=None, skiprows=None):
    return np.loadtxt(filename, usecols=usecols, delimiter=delimiter, dtype=dtype, skiprows=skiprows)


def naive_bayes_with_no_privacy():
    # URL of the dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # Local file name to save the data
    local_file_name = "adult.data"

    # Download the dataset if not already downloaded
    download_dataset(data_url, local_file_name)

    # Now load the dataset
    X_train = load_data(local_file_name, usecols=(0, 4, 10, 11, 12), delimiter=",")
    y_train = load_data(local_file_name, usecols=14, dtype=str, delimiter=",")
    X_test = load_data(local_file_name, usecols=(0, 4, 10, 11, 12), delimiter=",", skiprows=1)
    y_test = load_data(local_file_name, usecols=14, dtype=str, delimiter=",", skiprows=1)
    y_test = np.array(a[:-1] for a in y_test)

    # Train a regular (non-private) naive Bayes classifier and test its accuracy
    nonprivate_clf = GaussianNB
    nonprivate_clf.fit(X_train, y_train)



if __name__ == "__main__":
