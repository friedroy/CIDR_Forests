import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

save_p = 'visuals/evaluation/'
Path(save_p).mkdir(exist_ok=True, parents=True)


def att_residuals(true: np.ndarray, pred: np.ndarray, name: str=''):
    inds = np.argsort(true)
    resid = true[inds] - pred[inds]
    plt.figure()
    plt.plot([0, np.max(true)], [0, 0], '--k', lw=2)
    plt.scatter(true, resid, 10, alpha=.1)
    plt.xlabel('ndvi value')
    plt.ylabel('residuals')
    plt.savefig(save_p + name + 'ndvi_residuals.png')


def feature_residuals(X: np.ndarray, true: np.ndarray, pred: np.ndarray, features: list, name: str=''):
    for i, f in enumerate(features):
        inds = np.argsort(X[:, i])
        resid = true[inds] - pred[inds]
        plt.figure()
        plt.plot([np.min(X[:, i]), np.max(X[:, i])], [0, 0], '--k', lw=2)
        plt.scatter(X[inds, i], resid, 10, alpha=.1)
        plt.xlabel('{} value'.format(f))
        plt.ylabel('residuals')
        plt.savefig(save_p + name + '{}_residuals.png'.format(f))


def time_residuals(true: np.ndarray, pred: np.ndarray, times: np.ndarray, name: str=''):
    inds = np.argsort(times)
    resid = true[inds] - pred[inds]
    plt.figure()
    plt.plot([0, np.max(times)], [0, 0], '--k', lw=2)
    plt.scatter(times[inds], resid, 10, alpha=.1)
    plt.xlabel('year')
    plt.ylabel('residuals')
    plt.savefig(save_p + name + 'time_residuals.png')


def model_residuals(X: np.ndarray, y: np.ndarray, times: np.ndarray, features: list, models: list):
    for (name, mdl) in models:
        preds = mdl.predict(X)
        att_residuals(y, preds, name=name)
        feature_residuals(X, y, preds, features, name=name)
        time_residuals(y, preds, times, name=name)
        plt.close('all')
