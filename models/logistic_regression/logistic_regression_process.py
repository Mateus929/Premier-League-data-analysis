import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#load data
pd.set_option('future.no_silent_downcasting', True)
df = pd.read_csv('../../data/raw/matches.csv')
X = df[['poss', 'sot']].copy()
temp = df[['result']].copy()
y = temp.replace({'L':0, 'D':1, 'W':2})

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#save to raw files
X_test.to_csv('../../data/processed/logistic_regression/X_test.csv', index=False)
y_test.to_csv('../../data/processed/logistic_regression/y_test.csv', index=False)
X_train.to_csv('../../data/processed/logistic_regression/X_train.csv', index=False)
y_train.to_csv('../../data/processed/logistic_regression/y_train.csv', index=False)

#scater data
plt.figure(figsize=(12, 8))

for i in range(X_train.shape[0]):
    marker = 'x'
    if y_train.iloc[i].values[0] == 1:
        marker = 'o'
    elif y_train.iloc[i].values[0] == 2:
        marker = 's'
    plt.scatter(X_train.iloc[i]['poss'], X_train.iloc[i]['sot'], marker=marker, color='b')

plt.xlabel("Team's Possession", fontsize=12)
plt.ylabel("Team's shot on target", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('poss_and_shot_vs_result.png')