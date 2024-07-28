import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

big_mart_data = pd.read_csv('Train.csv')

# mean value of "Item_Weight" column
big_mart_data['Item_Weight'].mean()

# filling the missing values in "Item_weight column" with "Mean" value
big_mart_data['Item_Weight'].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

# mode of "Outlet_Size" column
big_mart_data['Outlet_Size'].mode()

# filling the missing values in "Outlet_Size" column with Mode
mode_of_Outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

miss_values = big_mart_data['Outlet_Size'].isnull()   

big_mart_data.loc[miss_values, 'Outlet_Size'] = big_mart_data.loc[miss_values,'Outlet_Type'].apply(lambda x: mode_of_Outlet_size[x])

# checking for missing values
big_mart_data.isnull().sum()

# label encoder
encoder = LabelEncoder()

big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

# splitting features and target
x = big_mart_data.drop(columns='Item_Outlet_Sales',axis=1)
y = big_mart_data['Item_Outlet_Sales'].values

# spliting the data into training data and testing data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print('x shape:...',x.shape)
print('x_train shape:...',x_train.shape)
print('x_test shape:...',x_test.shape)

# XGBoost Regressor
model = XGBRegressor()

model.fit(x_train,y_train)

# prediction on training data
training_data_prediction = model.predict(x_train)

# R squared value
r2_train = metrics.r2_score(y_train,training_data_prediction)
print('R squared value:...',r2_train)

# prediction on test data
test_data_prediction = model.predict(x_test)

# R squared value
r2_test = metrics.r2_score(y_test,test_data_prediction)
print('R squared value:...',r2_test)
