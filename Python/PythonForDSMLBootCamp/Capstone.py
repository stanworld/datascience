# USA 911 call

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("/home/stan/Downloads/Refactored_Py_DS_ML_Bootcamp-master/10-Data-Capstone-Projects/911.csv")
#df.info()
#df.describe()
x=df["zip"].value_counts().head(5)

x=df["twp"].value_counts().head(5)

x=df["title"].nunique()

df["Reason"]=df["title"].apply(lambda x: x.split(":")[0])
x=df["Reason"].value_counts().head(3)

import seaborn as sns
ax = sns.countplot(x="Reason", data=df,palette='viridis')
fig = ax.get_figure()
fig.savefig("output.png")

type(df["timeStamp"].iloc[0])
df["timeStamp"]=df["timeStamp"].apply(lambda x: pd.to_datetime(x))
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df["Day of Week"]=df["timeStamp"].apply(lambda x: x.weekday()).map(dmap)
df["Month"]=df["timeStamp"].apply(lambda x: x.month)

ax =sns.countplot(x="Day of Week",hue="Reason",data=df,palette="viridis")

plt.legend(bbox_to_anchor=(0.5,0.85), loc=2,borderaxespad=0)
fig = ax.get_figure()
fig.set_size_inches(11.7,8.27)
fig.savefig("output.png")

ax =sns.countplot(x="Month",hue="Reason",data=df,palette="viridis")

plt.legend(bbox_to_anchor=(0.5,0.85), loc=2,borderaxespad=0)
fig = ax.get_figure()
fig.set_size_inches(11.7,8.27)
fig.savefig("output.png")

byMonth=df.groupby(by="Month").count()

fig,ax = plt.subplots()

line1,=ax.plot(byMonth.index.get_level_values("Month"),byMonth["Reason"],label="Month to Reason")
line1.set_dashes([2, 2, 10, 2])
ax.legend()

sns.lmplot(x="Month",y="Reason",data=byMonth.reset_index())

df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

plt.figure(figsize=(12,6))
sns.heatmap(dayHour,cmap='viridis')

sns.clustermap(dayHour,cmap='viridis')


plt.show()





