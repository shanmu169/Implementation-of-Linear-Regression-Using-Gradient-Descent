# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start and import the required libraries: Pandas, NumPy, and Matplotlib.
2. Load the 50_Startups.csv dataset and extract the R&D Spend and Profit columns.
3. Normalize the input and output data by calculating their mean and standard deviation.
4. Initialize slope, intercept, learning rate, and epochs, then calculate gradients and update the slope and intercept using *Gradient Descent*.
5. Calculate the final slope and intercept, and predict the profit for an R&D expenditure of 100000.
6. Display the slope, intercept, and predicted profit, plot the regression line with actual data points, and stop.
## Program:
```
Program to implement the linear regression using gradient descent.
Developed by: Shanmuga priya K
RegisterNumber:  212225040401
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('/50_Startups.csv')

X = df['R&D Spend'].values
Y = df['Profit'].values

x_mean, x_std = X.mean(), X.std()
y_mean, y_std = Y.mean(), Y.std()

X = (X - x_mean) / x_std
Y = (Y - y_mean) / y_std

m = 0
b = 0
learning_rate = 0.01
iterations = 10000
n = len(X)

for i in range(iterations):
    Y_pred = m * X + b

    dm = (-2/n) * np.sum(X * (Y - Y_pred))
    db = (-2/n) * np.sum(Y - Y_pred)

    m = m - learning_rate * dm
    b = b - learning_rate * db

slope = m * y_std / x_std
intercept = y_mean + b * y_std - slope * x_mean

spend = 100000
profit = slope * spend + intercept

print("Slope:", round(slope, 2))
print("Intercept:", round(intercept, 2))
print("Predicted Profit:", round(profit, 2))

plt.scatter(df['R&D Spend'], df['Profit'],
            label='Actual Profit')

plt.plot(df['R&D Spend'],
         slope * df['R&D Spend'] + intercept,
         color='red', label='Regression Line')

plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.title('Simple Linear Regression using Gradient Descent')
plt.legend()
plt.show()

```

## Output:
<img width="1007" height="642" alt="image" src="https://github.com/user-attachments/assets/bec7ad82-2b94-4ca0-b8af-c5e9518b5185" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
