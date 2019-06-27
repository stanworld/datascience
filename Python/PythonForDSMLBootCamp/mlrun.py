from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
X,y = np.arange(10).reshape(5,2),list(range(5))

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("/home/stan/Downloads/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv")
df.info()
df.describe()
df.columns

x=sns.pairplot(df)
x.savefig("output.png")