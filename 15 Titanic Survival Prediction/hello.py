import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dastaset = pd.read_csv('train.csv')

# drop the 'Cabin' column from the dataframe
dastaset = dastaset.drop(columns='Cabin',axis=1)

# replacing the missing values in 'Age' column wiht mean value
dastaset['Age'].fillna(dastaset['Age'].mean(),inplace=True)

# finding the mode value of 'Embarked' column
dastaset['Embarked'].mode()
# replacing the missing values in 'Embarked' column with this mode value
dastaset['Embarked'].fillna(dastaset['Embarked'].mode()[0],inplace=True)

# converting cetagorical column titanic data
dastaset.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':1}},inplace=True)

x = dastaset.drop(columns=['PassengerId','Name','Ticket','Survived'],axis=1)
y = dastaset['Survived'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model = LogisticRegression()

# fit the model
model.fit(x_train,y_train)

# accuracy on training data
x_train_prediction = model.predict(x_train)
print('x_train prediction:...',x_train_prediction)

# training data accuracy
training_data_accuracy = accuracy_score(y_train,x_train_prediction)
print('Accuracy score on training data:...',training_data_accuracy)

# accuracy score on test data
x_test_prediction = model.predict(x_test)
print('x_test preduction:...',x_test_prediction)

# test data accuracy
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print('Accuracy score on test data:...',test_data_accuracy)
