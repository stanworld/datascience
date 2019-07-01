from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
X,y = np.arange(10).reshape(5,2),list(range(5))

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# df=pd.read_csv("/home/stan/Downloads/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/USA_Housing.csv")
# df.info()
# df.describe()
# df.columns

#x=sns.pairplot(df)
#x.savefig("output.png")

#x=sns.distplot(df['Price'])
#plt.show()


# X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
#                'Avg. Area Number of Bedrooms', 'Area Population']]
# y = df['Price']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)

predictions = lm.predict(X_test)
#plt.scatter(y_test,predictions)

#sns.distplot((y_test-predictions),bins=50)
#plt.show()

# logistic regression : for classification
train=pd.read_csv("/home/stan/Downloads/Refactored_Py_DS_ML_Bootcamp-master/13-Logistic-Regression/titanic_train.csv")
# use heatmap to find out which part of data is missing most
sns.heatmap(train.isnull(),yticklabels=False,cbar=False, cmap="viridis")
plt.show()
