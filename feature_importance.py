from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from load_data import load_csv_tensor, tensor_to_features
import numpy as np
import matplotlib.pyplot as plt

tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope'], return_years=True)
X, y, features = tensor_to_features(tens, f2i, lookback=1, remove_att=True)

chosen = np.random.choice(X.shape[0], int(np.ceil(X.shape[0]*.05)), replace=False)
inds = np.zeros(X.shape[0])
inds[chosen] = 1
inds = inds.astype(bool)
X_val, y_val = X[inds], y[inds]
X, y = X[~inds], y[~inds]
print('{} training samples\n {} test samples'.format(X.shape[0], X_val.shape[0]))

models = [
    ('Linear Regression', LinearRegression(normalize=True)),
    ('DT-5', DecisionTreeRegressor(max_depth=5)),
    ('DT-10', DecisionTreeRegressor(max_depth=10)),
    ('RF-5', RandomForestRegressor(max_depth=5)),
    ('RF-10', RandomForestRegressor(max_depth=10)),
    ('XGBoost-5', GradientBoostingRegressor(max_depth=5)),
    ('XGBoost-10', GradientBoostingRegressor(max_depth=10)),
]
for (name, model) in models:
    model = model.fit(X, y)
    score = model.score(X_val, y_val)
    print('{}: train {:.3f}, test {:3f}'.format(name, model.score(X, y), model.score(X_val, y_val)))

    res = permutation_importance(model, X_val, y_val, n_repeats=100)
    means = []
    stds = []
    for i in range(len(features)):
        means.append(max(res.importances_mean[i], 0))
        stds.append(res.importances_std[i] if res.importances_mean[i] > 0 else 0)
    inds = np.argsort(means)
    means, stds = np.array(means)[inds], np.array(stds)[inds]
    names = np.array(features)[inds]

    plt.figure()
    plt.barh(np.arange(len(names)), means, xerr=stds, capsize=5, tick_label=names)
    plt.title(name)
    plt.xlabel('perm. feature importance')
    plt.savefig('visuals/{}_fi.png'.format(name))
    plt.show()
