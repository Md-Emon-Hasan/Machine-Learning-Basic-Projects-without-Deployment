import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.linear_model import Lasso

car_dataset = pd.read_csv("car data.csv")

# checking the distribution of categorical data
fuel = car_dataset['Fuel_Type'].value_counts()
seller = car_dataset['Seller_Type'].value_counts()
transmission = car_dataset['Transmission'].value_counts()

# encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)
# encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)
# encoding "Transmission" Column
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)

x = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
y = car_dataset['Selling_Price'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

model = LinearRegression()

# fit the model
model.fit(x_train,y_train)

# prediction on training data
training_data_prediction = model.predict(x_train)

# prediction on test data
test_data_prediction = model.predict(x_test)

lasso_model = Lasso()

# fit the model
lasso_model.fit(x_train,y_train)

# prediction on training data
training_data_prediction = lasso_model.predict(x_train)

# predicton on test data
test_data_prediction = lasso_model.predict(x_test)

# visualize the actual prices and predicted prices
plt.scatter(y_test,test_data_prediction)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Price vs Predicted Prices')
plt.show()

# in this case we see the so much difference value on the actual price vs predicted price so this method are not applicable
