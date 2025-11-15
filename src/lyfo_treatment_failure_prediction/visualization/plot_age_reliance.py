import pandas as pd
import seaborn as sns

data = pd.read_csv("results/km_data_lyfo_FCR.csv")


data[["y_pred", "age_at_tx"]].corr()

data[["NCCN_IPI_diagnosis", "age_at_tx"]].corr()