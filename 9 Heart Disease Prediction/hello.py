import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('heart_disease_data.csv')
dataset['target'].value_counts()

x = dataset.drop(columns='target',axis=1)
y = dataset['target'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print('x_shape:...',x.shape)
print('x_train:...',x_train.shape)
print('x_test:...',x_test.shape)

# model selection
model = LogisticRegression()

# fit the training data
model.fit(x_train,y_train)

# accuracy on the training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
print('Accuracy score on Training data:...',training_data_accuracy)

# accuracy on the test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)
print('Accuracy score on the Test data:...',test_data_accuracy)

# training data score
score = model.score(x_train,y_train)
print('y_train score:...',score)

# test data score
score = model.score(x_test,y_test)
print('x_test score:...',score)

input_data = (41,0,1,130,204,0,0,172,0,1.4,2,0,2)

# changing the input to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reashape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The Person does not have a Heart Disease')
else:
    print('The Person has Heart Dieases')
