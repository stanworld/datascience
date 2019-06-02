import numpy as np
import pandas as pd

dataset = pd.read_csv('/home/stan/Code/datascience/DataSet/Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

# Handling missing data
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan,strategy='mean')
