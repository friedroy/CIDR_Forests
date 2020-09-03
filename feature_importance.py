from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


save_p = 'visuals/feature_importance/'
Path(save_p).mkdir(exist_ok=True, parents=True)


def perm_feature_importance(X, y, features, models: list, n_perms: int=100):
    for (name, model) in models:
        print('{}: test score {:.3f}'.format(name, model.score(X, y)))
        res = permutation_importance(model, X, y, n_repeats=n_perms)
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
        plt.savefig(save_p + '{}_fi.png'.format(name))
        plt.show()
