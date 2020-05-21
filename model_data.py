from load_data import make_dataframes, build_tensors, features_labels_split
from load_data import blocked_folds, reshape_for_optim
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_validate
import pickle


def custom_cv(inds: np.ndarray):
    i = 0
    while i <= np.max(inds):
        yield np.where(inds == i)[0], np.where(inds == i)[0]
        i = i + 1


def rmse(est, X, y):
    return np.sqrt(np.mean((est.predict(X) - y)**2))


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
# split into blocks
X, y, inds = blocked_folds(X, y, num_splits=12, spatial_boundary=10, temporal_boundary=1, sp_block_sz=10, t_block_sz=3)
# flatten into proper feature and label vectors; after this step, the data should be ready for training
X_fl, y_fl, inds = reshape_for_optim(X, y, inds)
print('# total samples = {}'.format(len(inds)))

hold_out = (X_fl[inds == np.max(inds)], y_fl[inds == np.max(inds)])
tX, ty, tinds = X_fl[inds != np.max(inds)], y_fl[inds != np.max(inds)], inds[inds != np.max(inds)]
print('# train samples = {}'.format(len(tinds)))
print('# test samples = {}'.format(len(hold_out[1])))

print('\nMean of true y values: {:.3f} +- {:.3f}'.format(np.mean(ty), np.std(ty)))

# returning the mean as the predicted value (as a baseline)
scores = np.sqrt(np.mean((np.mean(ty) - ty)**2))
print('\nReturning mean as prediction (RMSE): {:.3f}'.format(scores))

# simple linear regression
lin = LinearRegression()
scores = cross_val_score(lin, tX, ty, cv=custom_cv(tinds), scoring=rmse)
print('\nLinear regression validation (RMSE): ')
print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))

# RBF SVM regression
rbf_svr = SVR()
scores = cross_val_score(rbf_svr, tX, ty, cv=custom_cv(tinds), scoring=rmse)
print('\nRBF SVM regression validation (RMSE): ')
print('\t\tMean = {:.3f} +- {:.3f}'.format(np.mean(scores), np.std(scores)))
print('\t\tMax = {:.3f},  Min = {:.3f}'.format(np.max(scores), np.min(scores)))

# Decision Tree regression
dt = DecisionTreeRegressor()
scores = cross_val_score(dt, tX, ty, cv=custom_cv(tinds), scoring=rmse)
print('\nDecision Tree regression validation (RMSE): ')
print('\t\tMean = {:.7f} +- {:.7f}'.format(np.mean(scores), np.std(scores)))
print('\t\tMax = {:.7f},  Min = {:.7f}'.format(np.max(scores), np.min(scores)))

# todo test best model after validation on the held out data
# todo check how the models are affected by the block sizes and boundaries
# todo fix the manner in which the test samples are chosen
