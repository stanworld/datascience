import numpy as np
import pandas as pd

dataset = pd.read_csv('/home/stan/Code/datascience/DataSet/Data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

# Handling missing data
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan,strategy='mean')

def domainget(email):
    return email.split('@')[1]

x=domainget('user@domain.com')
print(x)

def containDog(input):
    return "dog" in input

x=containDog("good dog")
print(x)

def countDogs(input):
    count  = 0
    for word in input.lower().split():
        if word == 'dog':
            count +=1
    return count;

x=countDogs('how many dog in dog you dog')
print(x)

seq=['s1','s2','wer','tes']
x=list(filter(lambda x: x[0]=='s',seq))
print(x)

x=np.zeros(10)
print(x)

import numpy as np
import pandas as pd

outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index= list(zip(outside,inside))
hier_index= pd.MultiIndex.from_tuples(hier_index)

print(hier_index)

df = {'A':[1,2,np.nan],'B':[5,np.nan,np.nan],'C':[1,2,3]}
df =pd.DataFrame(df)

t=df.dropna(axis=1)
print(t)
