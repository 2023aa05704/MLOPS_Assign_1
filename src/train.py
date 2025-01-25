import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


data = pd.read_csv('data/housing.csv')

X = data.drop(['price', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus'], axis=1)
y = data.price.copy()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2, random_state=42)

n_estimators = 200
random_state = 42

model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

joblib.dump(model, 'model/rf-default.joblib')