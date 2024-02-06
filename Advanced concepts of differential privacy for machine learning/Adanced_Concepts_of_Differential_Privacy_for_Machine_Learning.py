import numpy as np
import diffprivlib.models as dp
import urllib.request
import os
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, r2_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.datasets import load_digits
from time import time


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
def load_data(filename, usecols, delimiter, dtype=None, skiprows=0):
    return np.loadtxt(filename, usecols=usecols, delimiter=delimiter, dtype=dtype, skiprows=skiprows)


def naive_bayes():
    # URL of the dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # Local file name to save the data
    local_file_name = "../Data/adult.data"

    # Download the dataset if not already downloaded
    download_dataset(data_url, local_file_name)

    # Now load the dataset
    X_train = load_data(local_file_name, usecols=(0, 4, 10, 11, 12), delimiter=",")
    y_train = load_data(local_file_name, usecols=14, dtype=str, delimiter=",")
    X_test = load_data(local_file_name, usecols=(0, 4, 10, 11, 12), delimiter=",", skiprows=1)
    y_test = load_data(local_file_name, usecols=14, dtype=str, delimiter=",", skiprows=1)
    # y_test = np.array([a[:-1] for a in y_test])

    # Train a regular (non-private) naive Bayes classifier and test its accuracy
    nonprivate_clf = GaussianNB()
    nonprivate_clf.fit(X_train, y_train)

    print("Non-private test accuracy: %.2f%%" %
          (accuracy_score(y_test, nonprivate_clf.predict(X_test)) * 100))

    # Train a differentially private naive Bayes
    epsilon = 1
    # Specify bounds (min, max) of each feature
    dp_clf = dp.GaussianNB(epsilon=epsilon)

    dp_clf.fit(X_train, y_train)

    print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" %
          (dp_clf.epsilon, accuracy_score(y_test, dp_clf.predict(X_test)) * 100))

    epsilon = 0.01
    dp_clf = dp.GaussianNB(epsilon=epsilon)

    dp_clf.fit(X_train, y_train)

    print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" %
          (dp_clf.epsilon, accuracy_score(y_test, dp_clf.predict(X_test)) * 100))


def logistic_regression():
    # URL of the dataset
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # Local file name to save the data
    local_file_name = "../Data/adult.data"

    # Download the dataset if not already downloaded
    download_dataset(data_url, local_file_name)

    # Now load the dataset
    X_train = load_data(local_file_name, usecols=(0, 4, 10, 11, 12), delimiter=",")
    y_train = load_data(local_file_name, usecols=14, dtype=str, delimiter=",")
    X_test = load_data(local_file_name, usecols=(0, 4, 10, 11, 12), delimiter=",", skiprows=1)
    y_test = load_data(local_file_name, usecols=14, dtype=str, delimiter=",", skiprows=1)

    lr = Pipeline(
        [
            ("scaler", MinMaxScaler()),
            ("clf", LogisticRegression(solver="lbfgs"))
        ]
    )

    # Train a regular (non-private) logistic regression classifier and test its accuracy
    lr.fit(X_train, y_train)
    print("Non-private test accuracy: %.2f%%" %
          (accuracy_score(y_test, lr.predict(X_test)) * 100))

    # Train a differentially private logistic regression
    epsilon = 1
    dp_lr = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", dp.LogisticRegression(epsilon=epsilon))
    ])

    dp_lr.fit(X_train, y_train)
    print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" %
          (dp_lr["clf"].epsilon, accuracy_score(y_test, dp_lr.predict(X_test)) * 100))

    epsilon = 0.01
    dp_lr = Pipeline([
        ("scaler", MinMaxScaler()),
        ("clf", dp.LogisticRegression(epsilon=epsilon))
    ])

    dp_lr.fit(X_train, y_train)
    print("Differentially private test accuracy (epsilon=%.2f): %.2f%%" %
          (dp_lr["clf"].epsilon, accuracy_score(y_test, dp_lr.predict(X_test)) * 100))


def linear_regression():
    dataset = datasets.load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
    print("Train examples: %d, Test examples: %d" % (X_train.shape[0], X_test.shape[0]))

    # Non-private baseline
    regr = LinearRegression()
    regr.fit(X_train, y_train)
    baseline = r2_score(y_test, regr.predict(X_test))
    print("Non-private baseline: %.2f" % baseline)

    # Differentially private linear regression
    epsilon = 1.0
    dp_regr = dp.LinearRegression(epsilon=epsilon)
    dp_regr.fit(X_train, y_train)
    print("R2 score for epsilon=%.2f: %.2f" % (dp_regr.epsilon,
                                               r2_score(y_test, dp_regr.predict(X_test))))


def k_means_clustering():
    X_digits, y_digits = load_digits(return_X_y=True)
    data = scale(X_digits)

    n_samples, n_features = data.shape
    n_digits = len(np.unique(y_digits))
    labels = y_digits
    sample_size = 1000

    print("n_digits: %d, \t n_samples %d, \t n_features %d" %
          (n_digits, n_samples, n_features))

    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\t\tsilhouette')
    # Bench k-means
    bench_k_means(KMeans(init="k-means++", n_clusters=n_digits, n_init=100), name="K-means++", data=data, labels=labels, sample_size=sample_size)

    bench_k_means(KMeans(init="random", n_clusters=n_digits, n_init=100), name="random", data=data, labels=labels, sample_size=sample_size)

    # Diffferentially private K-means
    # bench_k_means(dp.KMeans(epsilon=1.0, bounds=None, init="k-means++", n_clusters=n_digits, n_init=100), name="dp_k-means", data=data, labels=labels, sample_size=sample_size)
    bench_k_means(dp.KMeans(epsilon=1.0, bounds=None, n_clusters=n_digits),
                  name="dp_k-means", data=data, labels=labels,
                  sample_size=sample_size)


def bench_k_means(estimator, name, data, labels, sample_size):
    t0 = time()
    estimator.fit(data)

    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


if __name__ == "__main__":
    # Naive Bayes
    # naive_bayes()

    # Logistic Regression
    # logistic_regression()

    # Linear Regression
    # linear_regression()

    # K-means clustering
    k_means_clustering()

