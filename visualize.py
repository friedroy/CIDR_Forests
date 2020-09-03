from load_data import load_csv_tensor, tensor_to_features
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

save_p = 'visuals/'
Path(save_p).mkdir(exist_ok=True, parents=True)

# tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope', 'lon', 'lat'])
tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope'], return_years=True,
                                      remove_atts=['swe'])
X, y, features = tensor_to_features(tens, f2i, lookback=1, remove_att=True)

# nX = (X - np.min(X, axis=0)[None, :])/(np.max(X, axis=0)-np.min(X, axis=0))[None, :]
# ny = (y - np.min(y))/(np.max(y)-np.min(y))
#
# corrs = []
# for i, f in enumerate(features):
#     inds = np.random.choice(X.shape[0], int(.05*X.shape[0]), replace=False)
#     c = np.corrcoef(X[:, i], y)
#     corrs.append(c[0, 1])
#     plt.figure()
#     plt.title('correlation={:.3f}'.format(c[0, 1]))
#     plt.scatter(nX[inds, i], ny[inds], 10, alpha=.5)
#     plt.plot([0, 1], [0, 1], '--k', lw=2)
#     plt.xlabel(f)
#     plt.ylabel('ndvi future')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.savefig(save_p + f + '_ndvicorr.png')
# plt.show()
#
# inds = np.argsort(corrs)[::-1]
# corrs = np.array(corrs)[inds]
# names = np.array(features)[inds]
# plt.figure()
# plt.bar(np.arange(len(inds)), corrs)
# plt.ylabel('correlation with ndvi')
# plt.xticks(np.arange(len(inds)), names, rotation=45)
# plt.savefig(save_p + 'ndvicorr_bars.png')
# plt.show()
#
# plt.close('all')
#
# for i, f in enumerate(features):
#     plt.figure()
#     plt.hist(X[:, i])
#     plt.xlabel(f + ' values')
#     plt.ylabel('counts')
#     plt.savefig(save_p + f + '_hist.png')
# plt.show()
#
# for i, f in enumerate(list(f2i.keys())):
#     inds = np.random.choice(tens.shape[0], 25, replace=False)
#     plt.figure()
#     for j in inds:
#         plt.plot(years, tens[j, :, f2i[f]], lw=.5)
#     plt.plot(years, np.mean(tens[:, :, f2i[f]], axis=0), 'k', lw=2)
#     plt.xlabel('years')
#     plt.ylabel(f)
#     plt.savefig(save_p + f + '_line.png')
# plt.show()

X = np.concatenate([X, y[:, None]], axis=1)
features.append('ndvi')
corrs = np.eye(X.shape[1])
for i, f1 in enumerate(features):
    for j, f2 in enumerate(features):
        if i!=j: corrs[i, j] = np.corrcoef(X[:, i], X[:, j])[0, 1]
plt.figure(dpi=300)
plt.imshow(corrs)
plt.xticks(np.arange(len(features)), features, rotation=45)
plt.yticks(np.arange(len(features)), features)
plt.colorbar()
plt.savefig(save_p + 'heatmap.png')
plt.show()