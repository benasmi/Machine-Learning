import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
import math


data = pd.read_csv("train.csv", usecols=["MSZoning","MSSubClass", "LotArea", "LotShape", "Neighborhood", "OverallQual","ExterQual", "TotalBsmtSF", "KitchenQual", "1stFlrSF","GrLivArea","GarageArea","SalePrice"])
data = data.dropna()
header = list(data.columns)
categorical_cols = ["MSZoning","Neighborhood","ExterQual","KitchenQual", "LotShape"]

Y = data["SalePrice"]
X = data.drop(['SalePrice'], axis=1)
X = pd.get_dummies(data[categorical_cols])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


print(X_train)
print(y_train)



regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})

def plot_predictions(err, predictions_points, accuracy):
    actual = []
    predictions = []
    for p in predictions_points:
        actual.append(p[0])
        predictions.append(p[1])
    print(actual)
    print(predictions)
    plt.figure(figsize=(12,5))
    plt.xlabel("Error: " + str(err) + " | " + "Accuracy: " + str(accuracy))
    plt.plot(actual, color="blue", linewidth=0.5, label="Actual price")
    plt.plot(predictions, color='red', linewidth=0.5, label="Predicted price")
    plt.legend(loc="upper left")
    plt.show()

def tree_error(data):
    actual = list(data["Actual"])
    predicted = list(data["Predicted"])
    err = 0
    percentage = 0
    prediction_points = []
    for i in range(len(actual)):
        err = err + (actual[i] - predicted[i]) ** 2
        prediction_points.append((actual[i],predicted[i]))
        percentage = percentage + math.fabs((actual[i] - predicted[i])/actual[i] * 100)
    return math.sqrt(err/len(data)), prediction_points, (100-percentage/len(data))

err, predictions, acc = tree_error(df)

plot_predictions(err,predictions,acc)
print(err)
print(predictions)
print(acc)
