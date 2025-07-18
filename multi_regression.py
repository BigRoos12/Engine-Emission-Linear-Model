import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn.metrics import r2_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)



csv = pd.read_csv("FuelConsumptionCo2.csv")
print(csv.head())
print(csv.describe())

sel_csv = csv[["ENGINESIZE","CYLINDERS","CO2EMISSIONS","FUELCONSUMPTION_COMB"]]
print(sel_csv.describe())

x = sel_csv[["ENGINESIZE","CYLINDERS","FUELCONSUMPTION_COMB"]]
y = sel_csv[["CO2EMISSIONS"]]


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size= 0.2,
    train_size= 0.8,
    random_state= 42,
    shuffle= True
)


regr = linear_model.LinearRegression()
x_train = np.asanyarray(x_train)
y_train = np.asanyarray(y_train)
regr.fit(x_train,y_train)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ', regr.intercept_)


y_pred = regr.predict(x_test)

r2 = r2_score(y_test,y_pred)
print(round(r2,ndigits=2))
