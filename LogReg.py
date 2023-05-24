import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
X = np.array([[1, 3], [2, 6], [3, 9], [4, 12], [5, 15]])
y = np.array([0, 0, 1, 1, 1])

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X, y)
x1 = np.linspace(0, 6, 100)
x2 = np.linspace(0, 16, 100)
X1, X2 = np.meshgrid(x1, x2)
X_grid = np.c_[X1.ravel(), X2.ravel()]
y_proba = log_reg.predict_proba(X_grid)[:, 1]
y_proba = y_proba.reshape(X1.shape)

# Plot the Graph
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k')
plt.contourf(X1, X2, y_proba, cmap='bwr', alpha=0.4)
plt.colorbar(label='Probability')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression')
plt.show()