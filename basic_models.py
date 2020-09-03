import numpy as np
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from load_data import load_csv_tensor, tensor_to_features
import matplotlib.pyplot as plt
from pathlib import Path

save_p = 'visuals/'
Path(save_p).mkdir(exist_ok=True, parents=True)
scoring = 'r2'


def simple_cv(models, X, y, n_splits: int=12):
    print('RSquared scores:')
    res, names = [], []
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=n_splits, scoring='r2')
        res.append(cv_results['test_score'])
        names.append(name)
        print('\t{}: {:.3f} ± {:.3f}'.format(name, cv_results['test_score'].mean(), cv_results['test_score'].std()),
              flush=True)

    plt.boxplot(res, labels=names)
    plt.ylabel('R^2')
    plt.title('Algorithm Comparison')
    plt.savefig(save_p + 'kfold_cv.png')
    plt.show()


def timeseries_cv(models, X, y, n_years: int, max_train_size: int=15000):
    print('RSquared scores:')
    res, names = [], []
    for name, model in models:
        tscv = TimeSeriesSplit(n_splits=n_years - 1, max_train_size=max_train_size)
        cv_results = cross_validate(model, X, y, cv=tscv, scoring='r2')
        res.append(cv_results['test_score'][6 if n_years > 7 else 1:])
        names.append(name)
        print('\t{}: {:.3f} ± {:.3f}'.format(name, cv_results['test_score'].mean(), cv_results['test_score'].std()),
              flush=True)

    plt.boxplot(res, labels=names)
    plt.ylabel('R^2')
    plt.title('Algorithm Comparison')
    plt.savefig(save_p + 'timeseries_cv.png')
    plt.show()


def year_cs(years: np.ndarray, blocksz: int, pad: int):
    un = np.sort(np.unique(years))
    blocks = [[i for i in un[j:j+blocksz]] for j in range(0, len(un), blocksz+pad)]
    for k in range(len(blocks)):
        block = np.sum([years == j for j in blocks[k]], axis=0)
        test = np.where(block)[0]
        train = np.where(1-block)[0]
        yield train, test


def time_block_cv(models, X, y, years, blocksz: int, pad: int=2):
    print('RSquared scores:')
    res, names = [], []
    for name, model in models:
        cv_results = cross_validate(model, X, y, cv=year_cs(years, blocksz, pad), scoring='r2')
        res.append(cv_results['test_score'])
        names.append(name)
        print('\t{}: {:.3f} ± {:.3f}'.format(name, cv_results['test_score'].mean(), cv_results['test_score'].std()),
              flush=True)

    plt.boxplot(res, labels=names)
    plt.title('Algorithm Comparison')
    plt.ylabel('R^2')
    plt.savefig(save_p + 'blocked_cv.png')
    plt.show()


n_years = 33
tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope'], return_years=True)
X, y, features, tm = tensor_to_features(tens, f2i, lookback=1, remove_att=True, return_years=True)

N = tens.shape[0]*n_years
# tX, ty, tm = X[:N], y[:N], tm[:N]
tX, ty, tm = X[:], y[:], tm[:]
n_samples = 5000
inds = np.sort(np.random.choice(len(ty), n_samples, replace=False))
tX, ty, tm = tX[:], ty[:], tm[:]

print('# samples = {}'.format(X.shape[0]))
print('# train samples = {}'.format(tX.shape[0]))

print('\nMean of true y values: {:.3f} ± {:.3f}'.format(np.mean(ty), np.std(ty)))

models = [
    ('LR', LinearRegression(normalize=True)),
    # ('BRidge', BayesianRidge(normalize=True)),
    # ('KernelRidge', KernelRidge(kernel='poly', degree=2)),
    # ('LinSVR', SVR(kernel='linear')),
    # ('RBF-SVR', SVR(kernel='rbf')),
    ('DT-5', DecisionTreeRegressor(max_depth=5)),
    ('DT-10', DecisionTreeRegressor(max_depth=10)),
    ('RF-5', RandomForestRegressor(max_depth=5)),
    ('RF-10', RandomForestRegressor(max_depth=10)),
    ('XGB-5', GradientBoostingRegressor(max_depth=5)),
    ('XGB-10', GradientBoostingRegressor(max_depth=10)),
    # ('AdaBoost', AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), loss='exponential')),
    # ('Bagging', BaggingRegressor()),
    # ('Extra', ExtraTreesRegressor()),
]

simple_cv(models, tX, ty, n_splits=12)
timeseries_cv(models, tX, ty, n_years, max_train_size=5 * tens.shape[0])
time_block_cv(models, tX, ty, tm, blocksz=2, pad=2)
