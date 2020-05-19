from main import make_tensors
import numpy as np
import matplotlib.pyplot as plt
import pickle


def autocorr(x):
    sz = x.shape
    result = np.correlate(x.flatten(), x.flatten(), mode='full')
    return result[result.size//2:].reshape(sz)


# ts, st, dicts = make_tensors()
# np.save('timeseries_tensor.npy', ts)
# np.save('static_tensor.npy', st)
# with open('att_dicts.pkl', 'wb') as f: pickle.dump(dicts, f)

ts = np.load('timeseries_tensor.npy')
st = np.load('static_tensor.npy')
with open('att_dicts.pkl', 'rb') as f: ts_dict, st_dict = pickle.load(f)

normalization = np.zeros(ts.shape[-2:])
corr = np.zeros(ts.shape[-2:])
ndvi = ts[ts_dict['ndvi']]

for i in range(ndvi.shape[0]):
    im = ndvi[i]
    inds = ~np.isnan(im)
    corr[inds] += autocorr(im[inds])
    normalization[inds] += 1
    break

mean_corr = corr / normalization

