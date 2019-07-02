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
# data preprocessing
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

# replace age with median of each Pclass
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
# since there are too many missing values in Cabin, just drop the column directly
train.drop('Cabin',axis=1,inplace=True)
# drop a row still with missing values
train.dropna(inplace=True)

# Convert Categorical features

sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket','PassengerId'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)

X = train.drop('Survived',axis=1)
y = train['Survived']

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),
                                                    train['Survived'], test_size=0.30,
                                                    random_state=101)

logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)
print(classification_report(y_test,predictions))