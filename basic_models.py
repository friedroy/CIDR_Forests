import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from load_data import load_csv_tensor, tensor_to_features
import matplotlib.pyplot as plt

scoring = 'r2'
n_folds = 12
ho_rat = .5

tens, f2i, _, years = load_csv_tensor('data/test2.csv', stats=['aspect', 'slope'])
X, y, features = tensor_to_features(tens, f2i, lookback=1, remove_att=['swe', 'pdsi', 'pet'])
X = (X - np.mean(X, axis=0)[None, :])/np.std(X, axis=0)[None, :]
y = (y - np.mean(y))/np.std(y)

ho = np.random.choice(len(y), int(np.ceil(ho_rat*len(y))), replace=False)
hold_out = (X[ho], y[ho])

train_inds = np.ones(len(y)).astype(bool)
train_inds[ho] = False
tX, ty = X[train_inds], y[train_inds]
normX = (tX - np.min(tX, axis=0)[None, :])/(np.max(tX, axis=0)[None, :] - np.min(tX, axis=0)[None, :])

print('# train samples = {}'.format(tX.shape[0]))
print('# test samples = {}'.format(len(hold_out[1])))

print('\nMean of true y values: {:.3f} ± {:.3f}'.format(np.mean(ty), np.std(ty)))

models = [
    ('LR', LinearRegression(normalize=True)),
    ('BRidge', BayesianRidge(normalize=True)),
    # ('LinSVR', SVR(kernel='linear')),
    # ('RBF-SVR', SVR(kernel='rbf')),
    ('DT', DecisionTreeRegressor(max_depth=5)),
    ('RF', RandomForestRegressor(max_depth=5, n_estimators=50)),
    # ('XGB', GradientBoostingRegressor(n_estimators=50)),
    # ('KNN', KNeighborsRegressor()),
]

print('RSquared scores:')
res, names = [], []
for name, model in models:
    tscv = TimeSeriesSplit(n_splits=len(years)-1, max_train_size=int(.33*tX.shape[0]))
    cv_results = cross_validate(model, tX, ty, cv=tscv, scoring='r2')
    res.append(cv_results['test_score'])
    names.append(name)
    print('\t{}: {:.3f} ± {:.3f}'.format(name, cv_results['test_score'].mean(), cv_results['test_score'].std()),
          flush=True)

plt.boxplot(res, labels=names)
plt.title('Algorithm Comparison')
plt.show()