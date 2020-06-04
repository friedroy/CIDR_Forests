import numpy as np
import matplotlib.pyplot as plt
import pickle

ts = np.load('timeseries_tensor.npy')
st = np.load('static_tensor.npy')
with open('att_dicts.pkl', 'rb') as f: ts_dict, st_dict = pickle.load(f)

# simple plot to show all aggregated features as an image
h, w = 2, 5
t = 0
plt.figure(dpi=300)
for i, k in enumerate(list(ts_dict.keys())):
    plt.subplot(h, w, i+1)
    im = ts[ts_dict[k], t]
    im[np.isnan(im)] = np.min(im[~np.isnan(im)])
    im = (im - np.min(im))/(np.max(im) - np.min(im))
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title(k)

N = 10
x = np.random.choice(ts.shape[2], N, replace=False)
y = np.random.choice(ts.shape[3], N, replace=False)
t_arr = np.arange(ts.shape[1])
plt.figure()
for i, k in enumerate(list(ts_dict.keys())):
    plt.subplot(w, h, i+1)
    for j in range(len(x)):
        arr = ts[ts_dict[k], :, x[j], y[j]]
        plt.plot(t_arr[~np.isnan(arr)], arr[~np.isnan(arr)], lw=2)
    arr = np.mean(ts[ts_dict[k]], axis=(1, 2))
    arr[np.isnan(arr)] = np.min(arr[~np.isnan(arr)])
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

plt.show()