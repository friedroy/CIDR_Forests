from load_data import load_csv_tensor, tensor_to_features
import numpy as np
from matplotlib import pyplot as plt

# tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope', 'lon', 'lat'])
tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope', 'lon', 'lat'], return_years=True,
                                      remove_atts=['swe'])
X, y, features = tensor_to_features(tens, f2i, lookback=1, remove_att=True)

X = (X - np.min(X, axis=0)[None, :])/(np.max(X, axis=0)-np.min(X, axis=0))[None, :]
y = (y - np.min(y))/(np.max(y)-np.min(y))

corrs = []
for i, f in enumerate(features):
    inds = np.random.choice(X.shape[0], int(.05*X.shape[0]), replace=False)
    c = np.corrcoef(X[:, i], y)
    corrs.append(c[0, 1])
#     plt.figure()
#     plt.title('correlation={:.3f}'.format(c[0, 1]))
#     plt.scatter(X[inds, i], y[inds], 10, alpha=.5)
#     plt.plot([0, 1], [0, 1], '--k', lw=2)
#     plt.xlabel(f)
#     plt.ylabel('ndvi future')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
# plt.show()

inds = np.argsort(corrs)[::-1]
corrs = np.array(corrs)[inds]
names = np.array(features)[inds]
plt.figure()
plt.bar(np.arange(len(inds)), corrs)
plt.ylabel('correlation with ndvi')
plt.xticks(np.arange(len(inds)), names, rotation=45)
plt.show()

# for i, f in enumerate(features):
#     inds = np.random.choice(tens.shape[0], 25, replace=False)
#     plt.figure()
#     for j in inds:
#         plt.plot(years, tens[j, :, f2i[f]], lw=.5)
#     plt.plot(years, np.mean(tens[:, :, f2i[f]], axis=0), 'k', lw=2)
#     plt.xlabel('years')
#     plt.ylabel(f)
# plt.show()