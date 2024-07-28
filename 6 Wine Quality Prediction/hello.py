import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_dataset = pd.read_csv('winequality-red.csv')

# separate the data and Label
x = wine_dataset.drop('quality',axis=1).values
y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0).values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

model = RandomForestClassifier()
model.fit(x_train,y_train)

# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction,y_test)
print('Test data Accuracy:...',test_data_accuracy)

input_data = (7.4,0.25,0.29,2.2,0.054000000000000006,19.0,49.0,0.99666,3.4,0.76,10.9)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)
