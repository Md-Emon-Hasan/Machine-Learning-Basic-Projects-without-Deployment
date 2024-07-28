import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# loading the data to a data frame
data_frame = pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)

# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

# checking the distribution of target variables
data_frame['label'].value_counts()

data_frame.groupby('label').mean()

# separating the features and target
x = data_frame.drop(columns='label',axis=1)
y = data_frame['label']

# splitting the data into training data and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print('x shape:...',x.shape)
print('x_train shape:...',x_train.shape)
print('x_test shape:...',x_test.shape)

# logistic regression
model = LogisticRegression()

# training the logestic regression model
model.fit(x_train,y_train)

# accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train,x_train_prediction)
print('Accuracy on training data:...',training_data_accuracy)

# accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test,x_test_prediction)
print('Accuracy on training data:...',test_data_accuracy)

# building a predictive system
input_data = (13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)

# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predictig for one datapoint
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The Breast cancer is Maligant')
else:
    print('The Breast cancer is Benigin')