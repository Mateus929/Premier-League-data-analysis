from xxlimited_35 import error

import pandas as pd
import numpy as np
from gradient_descent_predict import get_prediction_vector

X_train = pd.read_csv('../../data/processed/gradient_descent/X_train.csv').to_numpy()
y_train = pd.read_csv('../../data/processed/gradient_descent/y_train.csv').values.flatten()

y_hat = get_prediction_vector(X_train)
err = y_hat - y_train

max_error = np.max(np.abs(err))
min_error = np.min(np.abs(err))
average_error = np.mean(np.abs(err))
print(f"Max Error: {max_error}")
print(f"Min Error: {min_error}")
print(f"Average Error: {average_error}")