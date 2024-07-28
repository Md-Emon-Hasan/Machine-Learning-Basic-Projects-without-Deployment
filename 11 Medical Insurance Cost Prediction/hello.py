import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('insurance.csv')

# encoding sex column
dataset.replace({'sex':{'male':0,'female':1}},inplace=True)
# encoding smoker column
dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)
# encoding region column
dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)

x = dataset.drop(columns='charges',axis=1).values
y = dataset['charges'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print('x shape',x.shape)
print('x_train shape',x_train.shape)
print('x_test shape',x_test.shape)

# linear model
model = LinearRegression()

# fit the data into the model
model.fit(x_train,y_train)

# r squared value for training data
training_data_prediction = model.predict(x_train)
r2_train = metrics.r2_score(y_train,training_data_prediction)
print('R squard value:...',r2_train)

# r squared value for testing data
test_data_prediction = model.predict(x_test)
r2_test = metrics.r2_score(y_test,test_data_prediction)
print('R squard value:...',r2_test)

input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

print('The insurance cost is USD:...', prediction[0])