import pandas as pd
import numpy as np


def cost_function(w, b, X_train, y_train):
    assert (X_train.shape[0] == y_train.shape[0])
    n = X_train.shape[0]
    f_wb = 0
    for i in range(n):
        f_wb_i = np.dot(w, X_train[i]) + b - y_train[i]
        f_wb_i = f_wb_i ** 2
        f_wb += f_wb_i
    f_wb /= 2 * n
    return f_wb

def gradient_function(w, b, X_train, y_train):
    n = X_train.shape[0]
    temp = np.dot(X_train, np.transpose(w))
    f_vector = temp - y_train
    dj_dw = np.dot(np.transpose(X_train), f_vector)
    dj_dw /= n

    dj_db = 0
    for i in range(n):
        dj_db_i = np.dot(w, X_train[i]) + b - y_train[i]
        dj_db += dj_db_i
    dj_db /= n

    return dj_dw, dj_db

def gradient_descent(X_train, y_train, learning_rate, max_iteration):
    p = X_train.shape[1]
    w, b = np.zeros(p), 0
    previous_cost = float('inf')
    for _ in range(max_iteration):
        dw, db = gradient_function(w, b, X_train, y_train)
        cur_cost = cost_function(w, b, X_train, y_train)
        if np.all(dw == 0) and db == 0:
            break
        previous_cost = cur_cost
        w -= learning_rate * dw
        b -= learning_rate * db
        if _ % 100 == 0:
            print(f"Iteration {_}: w = {w}, b = {b}, cost = {cur_cost}")
    return w, b


# Load data
X_train = pd.read_csv('../data/processed/gradient_descent/X_train.csv').to_numpy()
y_train = pd.read_csv('../data/processed/gradient_descent/y_train.csv').values.flatten()

w, b = gradient_descent(X_train, y_train, learning_rate=0.0007, max_iteration=10000)

print("Weights:", w)
print("Bias:", b)
print("Cost function value:", cost_function(w, b, X_train, y_train))
