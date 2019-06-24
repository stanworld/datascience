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

import cufflinks as cf
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
init_notebook_mode(connected=True)
cf.go_offline()
df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())

df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})

import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)

plotly.offline.iplot({
    "data": [go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": go.Layout(title="hello world")
},auto_play=True)



import plotly
import plotly.graph_objs as go

plotly.offline.plot({
    "data": [go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 1])],
    "layout": go.Layout(title="hello world")
}, auto_open=True)