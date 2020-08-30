import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from load_data import load_csv_tensor, tensor_to_features
import matplotlib.pyplot as plt

scoring = 'r2'
n_folds = 12
n_years = 20
tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope'], return_years=True)
X, y, features = tensor_to_features(tens, f2i, lookback=1, remove_att=True)
# X = (X - np.mean(X, axis=0)[None, :])/np.std(X, axis=0)[None, :]
# y = (y - np.mean(y))/np.std(y)

N = tens.shape[0]*n_years
tX, ty = X[:N], y[:N]

print('# samples = {}'.format(X.shape[0]))
print('# train samples = {}'.format(tX.shape[0]))

print('\nMean of true y values: {:.3f} ± {:.3f}'.format(np.mean(ty), np.std(ty)))

models = [
    ('LR', LinearRegression(normalize=True)),
    # ('BRidge', BayesianRidge(normalize=True, tol=10e-6)),
    # ('KernelRidge', KernelRidge(kernel='poly', degree=2)),
    # ('LinSVR', SVR(kernel='linear')),
    # ('RBF-SVR', SVR(kernel='rbf')),
    ('DT-5', DecisionTreeRegressor(max_depth=5)),
    ('DT-10', DecisionTreeRegressor(max_depth=10)),
    ('RF', RandomForestRegressor()),
    ('XGB', GradientBoostingRegressor()),
    ('KNN', KNeighborsRegressor(n_neighbors=5, weights='uniform')),
]

print('RSquared scores:')
res, names = [], []
for name, model in models:
    tscv = TimeSeriesSplit(n_splits=len(years)-1, max_train_size=15000)
    cv_results = cross_validate(model, X, y, cv=tscv, scoring='r2')
    res.append(cv_results['test_score'])
    names.append(name)
    print('\t{}: {:.3f} ± {:.3f}'.format(name, cv_results['test_score'].mean(), cv_results['test_score'].std()),
          flush=True)

plt.boxplot(res, labels=names)
plt.title('Algorithm Comparison')
plt.show()