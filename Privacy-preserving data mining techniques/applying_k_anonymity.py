import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from cn.protect import Protect
import ydata_profiling
sns.set(style="darkgrid")


def applying_k_anonymity():
    # Local file name
    local_file_name = "../Data/adult.csv"
    df = pd.read_csv(local_file_name)
    print(df.shape)
    print(df.head())

    # prot = Protect(df)
    # prot.itypes.age = "quasi"
    # prot.itypes.sex = "quasi"
    # prot.privacy_model.k = 5
    # prot_df = prot.protect()
    # prot_df

    # The library cn is not available, change to the ydata_profiling for the next
    prot = ydata_profiling.ProfileReport(df)
    # quasi_columns = ["age", "sex"]
    # for column in quasi_columns:
        # prot.set_variable(f"types.{column}.type", "quasi")
        # prot.description_set[column].type = "quasi"

    # k_value = 5
    # prot.set_variable("privacy.k", k_value)

    prot.to_file("data_report.html")





if __name__ == "__main__":
    applying_k_anonymity()
