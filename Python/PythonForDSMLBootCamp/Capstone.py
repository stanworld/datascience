import numpy as np
import pandas as pd
import matplotlib as plt

df=pd.read_csv("/home/stan/Downloads/Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects/911.csv")
df.info()

x=df["zip"].value_counts().head(5)

x=df["twp"].value_counts().head(5)

x=df["title"].nunique()


print(x)