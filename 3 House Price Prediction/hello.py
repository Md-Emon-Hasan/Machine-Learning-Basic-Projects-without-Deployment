import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

house_price_dataset = pd.read_csv('boston_house_prices.csv')

x = house_price_dataset.drop(['price'],axis=1)
y = house_price_dataset['price']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

print('x shape:...',x.shape)
print('x_train shape:...',x_train.shape)
print('x_test.shape:...',x_test.shape)

# loading the model
model = XGBRegressor()

# fit with XGBboosts model
model.fit(x_train,y_train)

# prediction on training data
training_data_prediction = model.predict(x_train)
print(training_data_prediction)

# R squared error
score_1 = metrics.r2_score(y_train,training_data_prediction)
print('R squared error:...',score_1)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(y_train,training_data_prediction)
print('Mean Absolute Error:...',score_2)

plt.scatter(y_train,training_data_prediction)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Price vs Preicted Price')
plt.show()

# accuracy for prediction on test data
test_data_prediction = model.predict(x_test)

# R squared error
score_1 = metrics.r2_score(y_test,test_data_prediction)

# Mean absolute error
score_2 = metrics.mean_absolute_error(y_test,test_data_prediction)

print('R squared error:...',score_1)
print('Mean absolute error:...',score_2)