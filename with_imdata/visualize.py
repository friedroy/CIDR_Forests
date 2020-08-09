import numpy as np
import matplotlib.pyplot as plt
import pickle
from load_data import features_labels_split

ts = np.load('timeseries_tensor.npy')
st = np.load('static_tensor.npy')
with open('att_dicts.pkl', 'rb') as f: ts_dict, st_dict = pickle.load(f)
t = 0


def pearson(x: np.ndarray, y: np.ndarray):
    norm = np.sqrt(np.sum((x-np.mean(x))**2))*np.sqrt(np.sum((y-np.mean(y))**2))
    return np.sum(((x - np.mean(x))*(y - np.mean(y))))/norm


# simple plot to show all aggregated features as an image
h, w = 2, 5
plt.figure(dpi=300)
for i, k in enumerate(list(ts_dict.keys())):
    plt.subplot(h, w, i+1)
    im = ts[ts_dict[k], t]
    im[np.isnan(im)] = np.min(im[~np.isnan(im)])
    im = (im - np.min(im))/(np.max(im) - np.min(im))
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title(k)

N = 25
x = np.random.choice(ts.shape[2], N, replace=False)
y = np.random.choice(ts.shape[3], N, replace=False)
t_arr = np.arange(ts.shape[1])
plt.figure()
for i, k in enumerate(list(ts_dict.keys())):
    plt.subplot(w, h, i+1)
    A = ts[ts_dict[k]]
    A = A[:, ~np.isnan(np.sum(A, axis=0))]
    for j in range(len(x)):
        arr = (ts[ts_dict[k], :, x[j], y[j]] - np.min(A))/ \
              (np.max(A) - np.min(A))
        plt.plot(t_arr[~np.isnan(arr)], arr[~np.isnan(arr)], lw=2)

    arr = np.mean((A - np.min(A))/ \
                   (np.max(A) - np.min(A)), axis=(1))
    # arr[np.isnan(arr)] = np.min(arr[~np.isnan(arr)])
    plt.plot(t_arr, arr, '--k')
    plt.title(k)


# plot of static features as full images
plt.figure(dpi=300)
for i, k in enumerate(list(st_dict.keys())):
    plt.subplot(1, 3, i+1)
    im = st[st_dict[k]]
    im[np.isnan(im)] = np.min(im[~np.isnan(im)])
    im = (im - np.min(im))/(np.max(im) - np.min(im))
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title(k)


X, y, names = features_labels_split(ts, st, ts_dict['ndvi'], ts_dict, st_dict, history=1, surrounding=0)
mat = X[t]
mat[np.isnan(mat)] = np.mean(mat[~np.isnan(mat)])

# pearson correlation between each pixel and the middle pixel
p = mat[mat.shape[0]//2, mat.shape[1]//2]
print(mat.shape[0]//2, mat.shape[1]//2)
corr = np.zeros(mat.shape[:2])
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        corr[i, j] = pearson(p, mat[i, j])
plt.figure()
plt.imshow(corr)
plt.colorbar()

# todo use Moran's I between the blocks to calculate correlation between the blocks

plt.show()
