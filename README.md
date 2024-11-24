# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices
<H3>NAME: KEERTHANA S</H3>
<H3>REGISTER NO.: 212223240070</H3>
<H3>EX. NO.2</H3>
<H3>DATE:</H3>

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Data Collection**:  
   - Import essential libraries like pandas, numpy, sklearn, matplotlib, and seaborn.  
   - Load the dataset using `pandas.read_csv()`.

2. **Data Preprocessing**:  
   - Address any missing values in the dataset.  
   - Select key features for training the models.  
   - Split the dataset into training and testing sets with `train_test_split()`.

3. **Linear Regression**:  
   - Initialize the Linear Regression model from sklearn.  
   - Train the model on the training data using `.fit()`.  
   - Make predictions on the test data using `.predict()`.  
   - Evaluate model performance with metrics such as Mean Squared Error (MSE) and the R² score.

4. **Polynomial Regression**:  
   - Use `PolynomialFeatures` from sklearn to create polynomial features.  
   - Fit a Linear Regression model to the transformed polynomial features.  
   - Make predictions and evaluate performance similar to the linear regression model.

5. **Visualization**:  
   - Plot the regression lines for both Linear and Polynomial models.  
   - Visualize residuals to assess model performance.

## Program:
```py
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML240EN-SkillsNetwork/labs/data/CarPrice_Assignment.csv")

# Select features and target variable
X = data[['enginesize']]  # Predictor
y = data['price']         # Target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Polynomial Regression (degree = 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate models
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_linear))
print("Linear Regression R^2 score:", r2_score(y_test, y_pred_linear))
print("Polynomial Regression MSE:", mean_squared_error(y_test, y_pred_poly))
print("Polynomial Regression R^2 score:", r2_score(y_test, y_pred_poly))

# Visualization: Linear Regression
plt.scatter(X_test, y_test, color='red', label='Actual Prices')
plt.plot(X_test, y_pred_linear, color='blue', label='Linear Regression')
plt.title('Linear Regression for Car Prices')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()

# Visualization: Polynomial Regression
plt.scatter(X_test, y_test, color='red', label='Actual Prices')
plt.plot(X_test, y_pred_poly, color='green', label='Polynomial Regression')
plt.title('Polynomial Regression for Car Prices')
plt.xlabel('Engine Size')
plt.ylabel('Price')
plt.legend()
plt.show()

```

## Output:
![image](https://github.com/user-attachments/assets/b5fe0c2b-8c65-4a6b-bf22-b01ad8bba46f)
![image](https://github.com/user-attachments/assets/98dfc5ae-a501-49c8-8951-2005a77473df)

## Result:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
