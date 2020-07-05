from load_data import make_dataframes, build_tensors, features_labels_split
from load_data import blocked_folds, reshape_for_optim
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_validate
import matplotlib.pyplot as plt
import pickle


def rmse(est, X, y):
    return np.sqrt(np.mean((est.predict(X) - y)**2))


def custom_cv(inds: np.ndarray):
    i = 0
    while i <= np.max(inds):
        yield np.where(inds != i)[0], np.where(inds == i)[0]
        i = i + 1


# load the data by uncommenting the following block
ts = np.load('timeseries_tensor.npy')
st = np.load('static_tensor.npy')
with open('att_dicts.pkl', 'rb') as f: ts_dict, st_dict = pickle.load(f)


def bound_effects():
    # create temporal-spatial feature and label tensors
    X, y, names = features_labels_split(ts, st, ts_dict['ndvi'], ts_dict, st_dict, history=1, surrounding=0)

    scoring = 'r2'
    sb_vals = [0, 10, 25, 50]
    tb_vals = [0, 1, 2, 3]
    sblk_vals = [5, 10, 25, 50]
    tblk_vals = [1, 2, 3]
    # sb_vals = [0, 1]
    # tb_vals = [0, 1]
    # sblk_vals = [5, 10]
    # tblk_vals = [1, 2]
    tens = np.zeros((len(sb_vals), len(tb_vals), len(sblk_vals), len(tblk_vals), 2))
    for i, sb in enumerate(sb_vals):
        for j, tb in enumerate(tb_vals):
            for k, sblk in enumerate(sblk_vals):
                for m, tblk in enumerate(tblk_vals):
                    print(sb, tb, sblk, tblk)
                    # split into blocks
                    Xspl, yspl, inds = blocked_folds(X, y, num_splits=12,
                                                     spatial_boundary=sb,
                                                     temporal_boundary=tb,
                                                     sp_block_sz=sblk,
                                                     t_block_sz=tblk)
                    # flatten into proper feature and label vectors
                    X_fl, y_fl, inds = reshape_for_optim(Xspl, yspl, inds)
                    print('# total samples = {}'.format(len(inds)))
                    print('# of samples per fold = {}'.format(
                        int(np.mean([len(inds[inds == a]) for a in np.unique(inds)]))))
                    chs = np.random.choice(X_fl.shape[0], min(X_fl.shape[0], 50000), replace=False)
                    X_fl, y_fl, inds = X_fl[chs], y_fl[chs], inds[chs]
                    # Decision Tree regression
                    dt = DecisionTreeRegressor(max_depth=5)
                    # dt = LinearRegression(normalize=True)
                    res = cross_validate(dt, X_fl, y_fl, cv=custom_cv(inds), scoring=scoring, return_estimator=False,
                                         return_train_score=True)
                    tscores, scores = res['train_score'], res['test_score']

                    tens[i, j, k, m, 0] = np.mean(tscores)
                    tens[i, j, k, m, 1] = np.mean(scores)

    train_tens = tens[:, :, :, :, 0]
    test_tens = tens[:, :, :, :, 1]
    names = ['sp. boundary', 'temp. boundary', 'sp. block sz', 'temp. block sz']
    x_vals = [sb_vals, tb_vals, sblk_vals, tblk_vals]
    plt.figure()
    for i in range(4):
        plt.subplot(2, 2, i+1)
        plt.errorbar(x_vals[i], np.mean(train_tens, axis=tuple([j for j in range(4) if i!=j])),
                     np.std(train_tens, axis=tuple([j for j in range(4) if i!=j])), lw=2, label='train',
                     capsize=5)
        plt.errorbar(x_vals[i], np.mean(test_tens, axis=tuple([j for j in range(4) if i != j])),
                     np.std(test_tens, axis=tuple([j for j in range(4) if i != j])), lw=2, label='test',
                     capsize=5)
        plt.xlabel(names[i])
        plt.ylabel('R2')
        if i==0: plt.legend()
    plt.show()


def feature_effects():
    scoring = 'r2'
    hist_vals = [1, 2, 3, 4]
    # hist_vals = [1, 2]
    surr_vals = [0, 1, 2, 3]
    # surr_vals = [0, 1]
    tens = np.zeros((len(hist_vals), len(surr_vals), 2))
    for i, hist in enumerate(hist_vals):
        for j, surr in enumerate(surr_vals):
                print(hist, surr)

                # create temporal-spatial feature and label tensors
                X, y, _ = features_labels_split(ts, st, ts_dict['ndvi'], ts_dict, st_dict, history=hist, surrounding=surr)

                # split into blocks
                Xspl, yspl, inds = blocked_folds(X, y, num_splits=12,
                                                 spatial_boundary=10,
                                                 temporal_boundary=2,
                                                 sp_block_sz=25,
                                                 t_block_sz=1)
                # flatten into proper feature and label vectors; after this step, the data should be ready for training
                X_fl, y_fl, inds = reshape_for_optim(Xspl, yspl, inds)
                print('# total samples = {}'.format(len(inds)))
                print('# of samples per fold = {}'.format(
                    int(np.mean([len(inds[inds == a]) for a in np.unique(inds)]))))
                chs = np.random.choice(X_fl.shape[0], min(X_fl.shape[0], 10000), replace=False)
                X_fl, y_fl, inds = X_fl[chs], y_fl[chs], inds[chs]
                # Decision Tree regression
                dt = DecisionTreeRegressor(max_depth=5)
                res = cross_validate(dt, X_fl, y_fl, cv=custom_cv(inds), scoring=scoring, return_estimator=False,
                                     return_train_score=True)
                tscores, scores = res['train_score'], res['test_score']
                tens[i, j, 0] = np.mean(tscores)
                tens[i, j, 1] = np.mean(scores)

    train_tens = tens[:, :, 0]
    test_tens = tens[:, :, 1]

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.errorbar(hist_vals, np.mean(train_tens, axis=1), np.std(train_tens, axis=1), lw=2, label='train', capsize=5)
    plt.errorbar(hist_vals, np.mean(test_tens, axis=1), np.std(test_tens, axis=1), lw=2, label='test', capsize=5)
    plt.xlabel('# years lookback')
    plt.ylabel('R2')
    plt.legend()

    surr_vals = (2*np.array(surr_vals) + 1)**2
    plt.subplot(1, 2, 2)
    plt.errorbar(surr_vals, np.mean(train_tens, axis=0), np.std(train_tens, axis=0), lw=2, label='train', capsize=5)
    plt.errorbar(surr_vals, np.mean(test_tens, axis=0), np.std(test_tens, axis=0), lw=2, label='test', capsize=5)
    plt.xlabel('spatial radius')
    plt.ylabel('R2')

    plt.figure()
    plt.imshow(train_tens, interpolation='nearest')
    plt.title('train R2')
    plt.ylabel('look-back')
    plt.yticks(np.arange(train_tens.shape[0]), hist_vals)
    plt.ylim([-.5, train_tens.shape[0] + .5])
    plt.xlabel('spatial radius')
    plt.xticks(np.arange(train_tens.shape[1]), surr_vals)
    plt.xlim([-.5, train_tens.shape[0] + .5])
    plt.colorbar()

    plt.figure()
    plt.imshow(test_tens, interpolation='nearest')
    plt.title('test R2')
    plt.ylabel('look-back')
    plt.yticks(np.arange(test_tens.shape[0]), hist_vals)
    plt.ylim([-.5, test_tens.shape[0] + .5])
    plt.xlabel('spatial radius')
    plt.xticks(np.arange(test_tens.shape[1]), surr_vals)
    plt.xlim([-.5, test_tens.shape[0] + .5])
    plt.colorbar()

    plt.show()


feature_effects()