import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('Diabetes.csv')

x = diabetes_dataset.drop(columns='Outcome',axis=1).values
y = diabetes_dataset['Outcome'].values

scaler = StandardScaler()

scaler.fit(x)
standaridized_data = scaler.transform(x)
x = standaridized_data

# test and training data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print('x_shape:...',x.shape)
print('x_train shape:...',x_train.shape)
print('x_test shape:...',x_test.shape)

classifier = svm.SVC(kernel='linear')

# fit the data into the model
classifier.fit(x_train,y_train)

# accuracy score on the training data
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction,y_train)
print('Accuracy score of the training data:...',training_data_accuracy)

# accuracy score on the test data
x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)
print('Accuracy score of the tasting data:...',test_data_accuracy)

input_data = (0,137,40,35,168,43.1,2.288,33)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardize the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The preson is diabetic')