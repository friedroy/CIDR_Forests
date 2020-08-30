from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from load_data import load_csv_tensor, tensor_to_features
import numpy as np
import matplotlib.pyplot as plt


tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope'], return_years=True)
X, y, features = tensor_to_features(tens, f2i, lookback=2, remove_att=True)

n_years = 20
N = tens.shape[0]*n_years
X_val, y_val = X[N:N+tens.shape[0]], y[N:N+tens.shape[0]]
X, y = X[:N], y[:N]

model = RandomForestRegressor().fit(X, y)
score = model.score(X_val, y_val)
print(score)

res = permutation_importance(model, X_val, y_val, n_repeats=30)
means = []
stds = []
for i in range(len(features)):
    means.append(res.importances_mean[i])
    stds.append(res.importances_std[i])
inds = np.argsort(means)[::-1]
means, stds = np.array(means)[inds], np.array(stds)[inds]
names = np.array(features)[inds]

plt.figure()
plt.bar(np.arange(len(features)), means, yerr=stds, capsize=5)
plt.xticks(np.arange(len(features)), names)
plt.ylabel('perm. feature importance')
plt.show()
