import numpy as np
import pandas as pd

def mse(y_true, y_pred):# MSE calc
    return np.mean((y_true - y_pred) ** 2)

def simple_linear_regression(X, y): # Linear Reg
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean) ** 2)
    w = numerator / denominator
    b = y_mean - w * x_mean
    return w, b

def multiple_linear_regression(X, y): # Multiple Reg
    X_b = np.c_[np.ones((len(X), 1)), X]  
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    b = theta[0]
    w = theta[1:]
    return w, b

df = pd.read_csv('data.csv')  # Load data

#Assume last col-target, others-features
X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values

split = int(len(df) * 0.8)  #Train,test
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

 #Linear reg with 1st feature only
w_s, b_s = simple_linear_regression(X_train[:, 0], y_train) 
y_pred_simple = w_s * X_test[:, 0] + b_s
mse_simple = mse(y_test, y_pred_simple)
                                                                 
# Multiple reg with all features
w_m, b_m = multiple_linear_regression(X_train, y_train)  
y_pred_multiple = X_test.dot(w_m) + b_m
mse_multiple = mse(y_test, y_pred_multiple)

print("Simple Linear Regression")
print("Coefficient:", w_s)
print("Intercept:", b_s)
print("Mean Squared Error:", mse_simple)
print()

print("Multiple Linear Regression")
print("Coefficients:", w_m)
print("Intercept:", b_m)
print("Mean Squared Error:", mse_multiple)
