import numpy as np
from sklearn.model_selection import cross_validate, TimeSeriesSplit
import matplotlib.pyplot as plt
from pathlib import Path

save_p = 'visuals/validation/'
Path(save_p).mkdir(exist_ok=True, parents=True)
scoring = 'r2'


def year_cv(years: np.ndarray, blocksz: int, pad: int):
    un = np.sort(np.unique(years))
    blocks = [[i for i in un[j:j+blocksz]] for j in range(0, len(un), blocksz+pad)]
    for k in range(len(blocks)):
        block = np.sum([years == j for j in blocks[k]], axis=0)
        test = np.where(block)[0]
        train = np.where(1-block)[0]
        yield train, test


def cross_val(models, X, y, cv, name):
    print('RSquared scores:')
    res, names = [], []
    ret_mdls = []
    for f, model in models:
        cv_results = cross_validate(model, X, y, cv=cv, scoring='r2', return_estimator=True)
        res.append(cv_results['test_score'][10:] if name == 'tscv' else cv_results['test_score'])
        names.append(f)
        print('\t{}: {:.3f} ± {:.3f}'.format(f, res[-1].mean(), res[-1].std()), flush=True)
        ret_mdls.append((f, cv_results['estimator'][np.argmax(cv_results['test_score'])]))

    plt.boxplot(res, labels=names, showfliers=False)
    plt.ylabel('R^2')
    plt.title('Algorithm Comparison')
    plt.savefig(save_p + '{}_cv.png'.format(name))
    plt.show()
    return ret_mdls


def validate_models(X, y, times, models: list, val_type='tscv'):
    print('# samples = {}'.format(X.shape[0]))
    print('# train samples = {}'.format(X.shape[0]))

    print('\nMean of true y values: {:.3f} ± {:.3f}'.format(np.mean(y), np.std(y)))
    assert type(val_type) == int or val_type == 'tscv' or val_type == 'blocked'
    ys = len(np.unique(times))
    if type(val_type)==int: return cross_val(models, X, y, val_type, '{}kfold'.format(val_type))
    elif val_type == 'tscv':
        tscv = TimeSeriesSplit(n_splits=ys-1, max_train_size=10*X.shape[0]//ys)
        return cross_val(models, X, y, tscv, 'tscv')
    else:
        return cross_val(models, X, y, year_cv(times, blocksz=2, pad=2), 'blocked')
