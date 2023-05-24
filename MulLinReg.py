import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

california = fetch_california_housing()
df = pd.DataFrame(data=california.data, columns=california.feature_names)
df['target'] = california.target
X = df.drop('target', axis=1)
y = df['target']

# Multiple Linear Regression
mul_lin_reg = LinearRegression()
mul_lin_reg.fit(X, y)
y_pred = mul_lin_reg.predict(X)

# Plotting the graph
plt.scatter(y, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted')
plt.show()

coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': reg.coef_})
print(coefficients)



