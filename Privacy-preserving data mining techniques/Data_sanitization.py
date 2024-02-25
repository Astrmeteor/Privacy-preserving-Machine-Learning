import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder
from dist_utils import best_fit_distribution
from dist_utils import plot_result


def data_sanitization():
    # Local file name
    local_file_name = "../Data/adult.csv"
    df = pd.read_csv(local_file_name)
    print(df.shape)
    print(df.head())

    # Suppression
    df.drop(columns=["fnlwgt", "relationship"], inplace=True)
    print(df.head())

    # Working with categorical values
    encoders = [(["sex"], LabelEncoder()), (["race"], LabelEncoder())]
    mapper = DataFrameMapper(encoders, df_out=True)
    new_cols = mapper.fit_transform(df.copy())
    new_cols_df = pd.DataFrame(new_cols, columns=["sex", "race"])
    df = df.drop(columns=["sex", "race"])
    df = pd.concat([df, new_cols_df], axis="columns")
    print(df.head())

    # Working with continuous values
    # Perturbation race and age
    categorical = "race"
    continuous = ["age"]
    unchanged = []
    for col in list(df):
        if (col not in categorical) and (col not in continuous):
            unchanged.append(col)

    best_distributions = []
    for col in continuous:
        data = df[col]
        best_dist_name, best_dist_params = best_fit_distribution(data, 500)
        best_distributions.append((best_dist_name, best_dist_params))

    gendf = perturb_data(df, unchanged, categorical, continuous, best_distributions, n=48842)
    print(gendf.head())


def perturb_data(df, unchanged_cols, categorical_cols, continuous_cols, best_distributions, n, seed=0):
    np.random.seed(seed)
    data = {}

    for col in categorical_cols:
        counts = df[col].value_counts()
        data[col] = np.random.choice(list(counts.index), p=(counts/len(df)).values, size=n)

    for col, bd in zip(continuous_cols, best_distributions):
        dist = getattr(scipy.stats, bd[0])
        data[col] = np.round(dist.rvs(size=n, *bd[1]))

    for col in unchanged_cols:
        data[col] = df[col]

    return pd.DataFrame(data, columns=unchanged_cols + categorical_cols + continuous_cols)


if __name__ == "__main__":
    data_sanitization()