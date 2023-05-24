import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 8, 18, 32, 50])
poly_features = PolynomialFeatures(degree=2)
X_poly = poly_features.fit_transform(X)

# Polynomial Regression
p_reg = LinearRegression()
p_reg.fit(X_poly, y)
y_pred = p_reg.predict(X_poly)

# Plotting the Graph
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.show()
