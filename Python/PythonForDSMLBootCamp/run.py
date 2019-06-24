import pandas as pd

sal = pd.read_csv("/home/stan/Code/datascience/DataSet/Salaries.csv");

x=sal.head()

#x=sal.info()

x=sal["BasePay"].mean();

x=sal["OvertimePay"].max();

x=sal.loc[sal["EmployeeName"]=="JOSEPH DRISCOLL"]

y=x["TotalPayBenefits"]

x=sal.loc[sal["TotalPayBenefits"].max()==sal["TotalPayBenefits"]]


x=sal.groupby("Year").mean()["BasePay"]

x=sal["JobTitle"].nunique()

x=sal["JobTitle"].value_counts().head(5)

x=sal.groupby("Year").get_group(2013)["JobTitle"].value_counts()
y=sum(x.loc[x==1])

x=sum(sal["JobTitle"].apply(lambda x: "Chief" in x))

print(x)

import numpy as np
from plotly import __version__
print(__version__)

import plotly
import plotly.graph_objs as go


plotly.offline.plot({
    "data": [go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": go.Layout(title="hello world")
}, auto_open=True)

import matplotlib.pyplot as plt

x=np.linspace(0,5,11)
y = x**2

fig = plt.figure()

axes = fig.add_axes([0.1,0.1,0.8,0.8])
axes.plot(x,y)
axes1 = fig.add_axes([0.2,0.4,0.3,0.4])
axes1.plot(x,y)
plt.show()