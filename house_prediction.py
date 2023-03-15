import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

house_price_dataset = sklearn.datasets.load_boston()
print(house_price_dataset)

#loading the dataset to a Pandas Dataframe
house_price_dataframe = pd.DataFrame(house_price_dataset.data,columns = house_price_dataset.feature_names)

house_price_dataframe.head()

# add the target (price) column to the Dataframe
house_price_dataframe['price']= house_price_dataset.target

house_price_dataframe.head()

house_price_dataframe.shape

#check for missing valuse
house_price_dataframe.isnull().sum()

house_price_dataframe.describe()

correlation = house_price_dataframe.corr()

#constructing a heatmap to understand the correlation
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f',annot=True,annot_kws={'size':8}, cmap= 'Blues')

X = house_price_dataframe.drop(['price'],axis = 1)
Y = house_price_dataframe['price']

print(X)
print(Y)

X_train , X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 2)
print(X.shape, X_train.shape, X_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train

# loading the model
model = XGBRegressor()

# training the model with X_trian
model.fit(X_train, Y_train)

# accuracy for prediction on traning data
training_data_prediction = model.predict(X_train)
print(training_data_prediction)

# R squared error
score_1 = metrics.r2_score(Y_train, training_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_train, training_data_prediction)

print("R squared error : ",score_1)
print("Mean absolute Error ", score_2)

plt.scatter(Y_train, training_data_prediction)
plt.xlabel("Actual Prices",color = 'blue')
plt.ylabel("Predicted Prices",color = 'red')
plt.title("Actual Price Vs Predicted Prices")
plt.show()

# accuracy for predicton on test data
test_data_prediction = model.predict(X_test)

# R squared error
score_1 = metrics.r2_score(Y_test, test_data_prediction)

# Mean Absolute Error
score_2 = metrics.mean_absolute_error(Y_test, test_data_prediction)

print("R squared error : ",score_1)
print("Mean absolute Error ", score_2)

house_price_dataset.data[0].reshape(1,-1)
##transformation of new data
scaler.transform(house_price_dataset.data[0].reshape(1,-1))

model.predict(scaler.transform(house_price_dataset.data[0].reshape(1,-1)))

import pickle
pickle.dump(model,open("xgbooost.pkl","wb"))
pickled_model = pickle.load(open("xgbooost.pkl","rb"))
pickled_model.predict(scaler.transform(house_price_dataset.data[0].reshape(1,-1)))