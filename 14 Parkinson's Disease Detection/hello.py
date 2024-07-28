import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('parkinsons.csv')

dataset['status'].value_counts()

update_dataset = dataset.drop(columns='name',axis=1)

update_dataset.groupby('status').mean()

x = update_dataset.drop(columns=['status'],axis=1)
y = update_dataset['status'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print('x shape:...',x.shape)
print('x_train shape:...',x_train.shape)
print('x_test shape:...',x_test.shape)

# data standarization
scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# support vector machine model
model = svm.SVC()

# training the SVM model with training data
model.fit(x_train,y_train)

# accuracy score on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train,x_train_prediction)
print('Accuracy score on training data:...',training_data_accuracy)

# accuracy score on testing data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print('Accuracy score of test data:...',test_data_accuracy)

# building a predictive system
input_data = (197.07600,206.89600,192.05500,0.00289,0.00001,0.00166,0.00168,0.00498,0.01098,0.09700,0.00563,0.00680,0.00802,0.01689,0.00339,26.77500,0.422229,0.741367,-7.348300,0.177551,1.743867,0.085569)

# changing input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standarized the input_data
std_data = scaler.transform(input_data_reshaped)

prediction = model.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The person does not have Parkinsons Disease')
else:
    print('The Person has Parkinsons')
