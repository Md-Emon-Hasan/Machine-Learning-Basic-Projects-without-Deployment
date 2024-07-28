import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

dataset = pd.read_csv('gld_price_data.csv')

x = dataset.drop(['Date','GLD'],axis=1)
y = dataset['GLD']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)

model = RandomForestRegressor(n_estimators=100)

# training the model
model.fit(x_train,y_train)

# prediction on Test Data
test_data_prediction = model.predict(x_test)
print('Test data prediction:...',test_data_prediction)

# R squared error
error_score = metrics.r2_score(y_test,test_data_prediction)
print('R squared error:...',error_score)

y_test = list(y_test)

plt.plot(y_test,color='red',label='Actual Value')
plt.plot(test_data_prediction,color='blue',label='Actual Value')

plt.title('Actual Price vs Predicted Price')
plt.xlabel('Number of values')
plt.ylabel('GLD Price')
plt.legend()
plt.show()