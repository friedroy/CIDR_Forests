import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from load_data import load_csv_tensor, tensor_to_features
import matplotlib.pyplot as plt

cmap = plt.get_cmap("tab10")
scoring = 'r2'
n_folds = 12

lookback = [i+1 for i in range(5)]
models = [
    ('LR', LinearRegression(normalize=True)),
    ('DT-5', DecisionTreeRegressor(max_depth=5))
]
res = np.zeros((len(lookback), len(models), n_folds))
for i, l in enumerate(lookback):
    tens, f2i, _, years= load_csv_tensor('data/test2.csv', stats=['aspect', 'slope', 'lat', 'lon'], return_years=True)
    # tens, f2i, _, years = load_csv_tensor('data/test2.csv', stats=[], return_years=True)
    X, y, features = tensor_to_features(tens, f2i, lookback=l, remove_att=True)
    print('{} lookback years, {} samples, {} feats:'.format(l, X.shape[0], X.shape[1]))
    for j, (name, model) in enumerate(models):
        cv_results = cross_validate(model, X, y, cv=n_folds, scoring='r2')
        print('\t{}: {:.3f} ± {:.3f}'.format(name, cv_results['test_score'].mean(), cv_results['test_score'].std()),
              flush=True)
        res[i, j] = cv_results['test_score']

plt.figure()
for i, (n, _) in enumerate(models):
    m = np.mean(res[:, i, :], axis=1)
    s = np.std(res[:, i, :], axis=1)
    plt.fill_between(lookback, m-s, m+s, color=cmap(i), alpha=.1)
    plt.plot(lookback, m, lw=2, color=cmap(i), label=n)
plt.xlabel('# look-back years')
plt.ylabel('R2')
plt.legend()
plt.show()