import numpy as np
import pandas as pd
import matplotlib as plt

df=pd.read_csv("/home/stan/Downloads/Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects/911.csv")
#df.info()
#df.describe()
x=df["zip"].value_counts().head(5)

x=df["twp"].value_counts().head(5)

x=df["title"].nunique()

df["Reason"]=df["title"].apply(lambda x: x.split(":")[0])
x=df["Reason"].value_counts().head(3)

import seaborn as sns
ax = sns.countplot(x="Reason", data=df)
fig = ax.get_figure()
fig.savefig("output.png")

print(type(df["timeStamp"].iloc[0]))
df["timeStamp"]=df["timeStamp"].apply(lambda x: pd.to_datetime(x))
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df["Day of Week"]=df["timeStamp"].apply(lambda x: x.weekday()).map(dmap)

print(df["Day of Week"])