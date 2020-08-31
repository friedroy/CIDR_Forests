import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from load_data import load_csv_tensor, tensor_to_features
import matplotlib.pyplot as plt

scoring = 'r2'


def simple_cv(models, X, y, n_years: int):
    print('RSquared scores:')
    res, names = [], []
    for name, model in models:
        tscv = TimeSeriesSplit(n_splits=n_years - 1, max_train_size=15000)
        cv_results = cross_validate(model, X, y, cv=tscv, scoring='r2')
        res.append(cv_results['test_score'])
        names.append(name)
        print('\t{}: {:.3f} ± {:.3f}'.format(name, cv_results['test_score'].mean(), cv_results['test_score'].std()),
              flush=True)

    plt.boxplot(res, labels=names)
    plt.ylabel('R^2')
    plt.title('Algorithm Comparison')
    plt.show()


def year_cs(years: np.ndarray, n_splits: int, pad: int):
    un = np.sort(np.unique(years))
    jmp = int(np.ceil(len(un))/n_splits)
    pad = 0 if jmp-pad <= 0 else pad
    blocks = [[i for i in un[j:j+jmp-pad]] for j in range(0, len(un), jmp)]
    for k in range(len(blocks)):
        block = np.sum([years == j for j in blocks[k]], axis=0)
        test = np.where(block)[0]
        train = np.where(1-block)[0]
        yield train, test


def time_block_cv(models, X, y, years, n_splits: int, pad: int=2):
    print('RSquared scores:')
    res, names = [], []
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=year_cs(years, n_splits, pad), scoring='r2')
        res.append(cv_results['test_score'])
        names.append(name)
        print('\t{}: {:.3f} ± {:.3f}'.format(name, cv_results['test_score'].mean(), cv_results['test_score'].std()),
              flush=True)

    plt.boxplot(res, labels=names)
    plt.title('Algorithm Comparison')
    plt.ylabel('R^2')
    plt.show()


n_years = 25
tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope'], return_years=True)
X, y, features, tm = tensor_to_features(tens, f2i, lookback=1, remove_att=True, return_years=True)

N = tens.shape[0]*n_years
tX, ty, tm = X[:N], y[:N], tm[:N]
n_samples = int(10e3)
inds = np.sort(np.random.choice(len(ty), n_samples, replace=False))
tX, ty, tm = tX[inds], ty[inds], tm[inds]

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
]

# simple_cv(models, tX, ty, n_years)
time_block_cv(models, tX, ty, tm, 10, pad=1)
