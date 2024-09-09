import numpy as np

# Values are generated from "gradient_descent.py", we cen run the program here as well.
CONST_W = np.array([0.0175366,  0.35136905])
CONST_B = 0.0013965657016038466

def predict(x_i, w=CONST_W, b=CONST_B):
    return np.dot(x_i, w) + b

def get_prediction_vector(X_train, w=CONST_W, b=CONST_B):
    n = X_train.shape[0]
    y_hat = np.zeros(n)
    for i in range(n):
        y_hat[i] = predict(X_train[i], w, b)
    return y_hat
