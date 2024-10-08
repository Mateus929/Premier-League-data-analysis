import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
df = pd.read_csv('../../data/raw/matches.csv')

# Load input data
df['100_minus_poss'] = 100 - df['poss']
X1 = df[['poss', 'gf']].copy()
X1['goal'] = X1['gf']
X2 = df[['100_minus_poss', 'ga']].copy()
X2['poss'] = X2['100_minus_poss']
X2['goal'] = X2['ga']
X1.drop(columns='gf', inplace=True)
X2.drop(columns='ga', inplace=True)
X2.drop(columns='100_minus_poss', inplace=True)
X = pd.concat([X1, X2], axis=0)

# Load output data
y1 = df[['xg']]
y2 = df[['xga']].copy()
y2['xg'] = y2['xga']
y2.drop(columns='xga', inplace=True)
y = pd.concat([y1, y2], axis=0)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#save to raw files
X_test.to_csv('../../data/processed/gradient_descent/X_test.csv', index=False)
y_test.to_csv('../../data/processed/gradient_descent/y_test.csv', index=False)
X_train.to_csv('../../data/processed/gradient_descent/X_train.csv', index=False)
y_train.to_csv('../../data/processed/gradient_descent/y_train.csv', index=False)




fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_train['goal'], X_train['poss'], y_train['xg'], c='blue', marker='o')
ax.set_xlabel('Goals', fontsize=12)
ax.set_ylabel('Possession', fontsize=12)
ax.set_zlabel('Expected Goals (xG)', fontsize=12)
ax.set_title('3D Scatter Plot of Goals, Possession, and xG', fontsize=14)
plt.tight_layout()
plt.savefig('3d_scatter_plot.png')
print("3D Scatter plot saved as '3d_scatter_plot.png'")

