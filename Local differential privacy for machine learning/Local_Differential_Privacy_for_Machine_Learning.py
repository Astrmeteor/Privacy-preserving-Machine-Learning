import numpy as np
import os
import urllib.request

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


if __name__ == "__main__":
    playing_with_the_US_census_dataset()