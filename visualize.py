from load_data import load_csv_tensor, tensor_to_features
import numpy as np
from matplotlib import pyplot as plt

# tens, f2i, _, years = load_csv_tensor('data/test2.csv', stats=['aspect', 'slope', 'lon', 'lat'])
tens, f2i, _, years = load_csv_tensor('data/test2.csv', stats=['slope'])
X, y, features = tensor_to_features(tens, f2i, lookback=1, remove_att=['swe', 'pdsi', 'pet'])

X = (X - np.min(X, axis=0)[None, :])/(np.max(X, axis=0)-np.min(X, axis=0))[None, :]
y = (y - np.min(y))/(np.max(y)-np.min(y))

for i, f in enumerate(features):
    inds = np.random.choice(X.shape[0], int(.05*X.shape[0]), replace=False)
    c = np.corrcoef(X[:, i], y)
    plt.figure()
    plt.title('correlation={:.3f}'.format(c[0, 1]))
    plt.scatter(X[inds, i], y[inds], 10, alpha=.5)
    plt.plot([0, 1], [0, 1], '--k', lw=2)
    plt.xlabel(f)
    plt.ylabel('ndvi future')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
plt.show()

for i, f in enumerate(features):
    inds = np.random.choice(tens.shape[0], 25, replace=False)
    plt.figure()
    for j in inds:
        plt.plot(years, tens[j, :, f2i[f]], lw=.5)
    plt.plot(years, np.mean(tens[:, :, f2i[f]], axis=0), 'k', lw=2)
    plt.xlabel('years')
    plt.ylabel(f)
plt.show()