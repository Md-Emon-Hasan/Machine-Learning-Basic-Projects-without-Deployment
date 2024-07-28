import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

calories_data = pd.read_csv('calories.csv')
exercise_data = pd.read_csv('exercise.csv')

calories_data = pd.concat([exercise_data,calories_data['Calories']],axis=1)

calories_data.replace({'Gender':{'male':0,'female':1}},inplace=True)

x = calories_data.drop(columns=['User_ID','Calories'],axis=1)
y = calories_data['Calories']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print('x shape:...',x.shape)
print('x_train shape:...',x_train.shape)
print('x_test.shape:...',x_test.shape)

# loading the model
model = XGBRegressor()

# training the model with x_train
model.fit(x_train,y_train)

# prediction on test data
test_data_prediction = model.predict(x_test)
print('test_data_prediction:...',test_data_prediction)

# mean absolute error
mean = metrics.mean_absolute_error(y_test,test_data_prediction)
print('Mean Absolute Error:...',mean)
