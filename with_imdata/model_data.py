from load_data import make_dataframes, build_tensors, features_labels_split
from load_data import blocked_folds, reshape_for_optim
import numpy as np
from sklearn.linear_model import LinearRegression, TweedieRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, cross_validate
import matplotlib.pyplot as plt
import pickle


def rmse(est, X, y):
    return np.sqrt(np.mean((est.predict(X) - y)**2))


rot = 90
capsize = 3
scoring = 'r2'


def custom_cv(inds: np.ndarray):
    i = 0
    while i <= np.max(inds):
        yield np.where(inds != i)[0], np.where(inds == i)[0]
        i = i + 1


def trees_feature_importance(mdl_list: list, feature_names: list, top_n: int=10):
    importances = np.mean([t.feature_importances_ for t in mdl_list], axis=0)
    std = np.std([t.feature_importances_ for t in mdl_list], axis=0)
    inds = np.argsort(importances)[::-1]
    plt.figure()
    plt.title('Decision Tree Feature Importance')
    plt.bar(np.arange(len(importances))[:top_n], importances[inds][:top_n],
            yerr=std[inds][:top_n], capsize=capsize, align='center')
    plt.xticks(np.arange(len(importances))[:top_n], feature_names[:top_n], rotation=rot)
    plt.show()


def forest_feature_importance(mdl_list: list, feature_names: list, top_n: int=10):
    importances = np.mean([f.feature_importances_ for f in mdl_list], axis=0)
    std = np.std([t.feature_importances_ for f in mdl_list for t in f.estimators_], axis=0)
    inds = np.argsort(importances)[::-1]
    plt.figure()
    plt.title('Random Forest Feature Importance')
    plt.bar(np.arange(len(importances))[:top_n], importances[inds][:top_n],
            yerr=std[inds][:top_n], capsize=capsize, align='center')
    plt.xticks(np.arange(len(importances))[:top_n], feature_names[:top_n], rotation=rot)
    plt.show()


# load data and create tensors
# df, fdf = make_dataframes()
# df = pd.read_pickle('df.pkl')
# fdf = pd.read_pickle('fdf.pkl')
# ts, st, (ts_dict, st_dict) = build_tensors(df, fdf)

# save the tensors to remove redundant recomputations each run
# np.save('timeseries_tensor.npy', ts)
# np.save('static_tensor.npy', st)
# with open('att_dicts.pkl', 'wb') as f: pickle.dump((ts_dict, st_dict), f)


# load the data by uncommenting the following block
ts = np.load('timeseries_tensor.npy')
st = np.load('static_tensor.npy')
with open('att_dicts.pkl', 'rb') as f: ts_dict, st_dict = pickle.load(f)

# create temporal-spatial feature and label tensors
X, y, names = features_labels_split(ts, st, ts_dict['ndvi'], ts_dict, st_dict, history=1, surrounding=0)
names = [n.replace('_', ' : ') for n in names]
# split into blocks
X, y, inds = blocked_folds(X, y, num_splits=12, spatial_boundary=10, temporal_boundary=1, sp_block_sz=20, t_block_sz=3)
# flatten into proper feature and label vectors; after this step, the data should be ready for training
X_fl, y_fl, inds = reshape_for_optim(X, y, inds)
print('# total samples = {}'.format(len(inds)))

# normalize data (essential for SVM and linear regression)
X_fl = (X_fl - np.min(X_fl, axis=0))/(np.max(X_fl, axis=0) - np.min(X_fl, axis=0))

ho = np.random.choice(len(y_fl), int(np.ceil(.1*len(y_fl))), replace=False)
hold_out = (X_fl[ho], y_fl[ho])

train_inds = np.ones(len(y_fl)).astype(bool)
train_inds[ho] = False
tX, ty, tinds = X_fl[train_inds], y_fl[train_inds], inds[train_inds]
print('# train samples = {}'.format(len(tinds)))
print('# test samples = {}'.format(len(hold_out[1])))
print('# of samples per fold = {}'.format(int(np.mean([len(inds[inds == a]) for a in np.unique(inds)]))))

print('\nMean of true y values: {:.3f} +- {:.3f}'.format(np.mean(ty), np.std(ty)))

# returning the mean as the predicted value (as a baseline)
scores = np.sqrt(np.mean((np.mean(ty) - ty)**2))
print('\nReturning mean as prediction - train RMSE: {:.3f}'.format(scores))

# simple linear regression
lin = LinearRegression(normalize=True)
res = cross_validate(lin, tX, ty, cv=custom_cv(tinds), scoring=scoring, return_estimator=True)
scores, estims = res['test_score'], res['estimator']
print('\nLinear regression validation (R2): ')
print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))
mdl = estims[np.argmin(scores)]
print('\t\tWeights for each feature:',
      [(names[i], np.round(mdl.coef_[i], 2)) for i in np.argsort(np.abs(mdl.coef_))[::-1]])

# Normal GLM regression
glmn = TweedieRegressor(power=0, alpha=0)
res = cross_validate(glmn, tX, ty, cv=custom_cv(tinds), scoring=scoring, return_estimator=True)
scores, _ = res['test_score'], res['estimator']
print('\nNormal GLM regression validation (R2): ')
print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))

# # linear SVM regression
# lin_svr = SVR(kernel='linear')
# res = cross_validate(lin_svr, tX, ty, cv=custom_cv(tinds), scoring=scoring, return_estimator=True)
# scores, estims = res['test_score'], res['estimator']
# print('\nLinear SVM regression validation (R2): ')
# print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
# print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))
# mdl = estims[np.argmin(scores)]
# print('\t\tWeights for each feature:',
#       [(names[i], np.round(mdl.coef_[0][i], 2)) for i in np.argsort(mdl.coef_[0])[::-1]])
#
# # RBF SVM regression
# rbf_svr = SVR()
# res = cross_validate(rbf_svr, tX, ty, cv=custom_cv(tinds), scoring=scoring, return_estimator=True)
# scores, _ = res['test_score'], res['estimator']
# print('\nRBF SVM regression validation (R2): ')
# print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
# print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))

# Decision Tree regression
dt = DecisionTreeRegressor(max_depth=5)
res = cross_validate(dt, tX, ty, cv=custom_cv(tinds), scoring=scoring, return_estimator=True)
scores, estims = res['test_score'], res['estimator']
print('\nDecision Tree regression validation (R2): ')
print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))
trees_feature_importance(estims, names)

# Decision Tree Friedman_MSE regression
dtf = DecisionTreeRegressor(criterion='friedman_mse')
res = cross_validate(dtf, tX, ty, cv=custom_cv(tinds), scoring=scoring, return_estimator=True)
scores, estims = res['test_score'], res['estimator']
print('\nDecision Tree with Friedman MSE regression validation (R2): ')
print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))
trees_feature_importance(estims, names)

# Random Forest regression
rf = RandomForestRegressor(max_depth=5, n_estimators=15)
res = cross_validate(rf, tX, ty, cv=custom_cv(tinds), scoring=scoring, return_estimator=True)
scores, estims = res['test_score'], res['estimator']
print('\nRandom Forest regression validation (R2): ')
print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))
forest_feature_importance(estims, names)

# Gradient Boosting regression
gb = GradientBoostingRegressor(n_estimators=15)
res = cross_validate(gb, tX, ty, cv=custom_cv(tinds), scoring=scoring, return_estimator=True)
scores, estims = res['test_score'], res['estimator']
print('\nGradient Boosting regression validation (R2): ')
print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))
forest_feature_importance(estims, names)

# todo test best model after validation on the held out data
# todo check how the models are affected by the block sizes and boundaries
