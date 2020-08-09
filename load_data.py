import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


# possible aggregations (averaging, summing, minimum and maximum)
avg_func = lambda x: np.mean(x, axis=0)
sum_func = lambda x: np.sum(x, axis=0)
min_func = lambda x: np.nanmin(x, axis=0)
max_func = lambda x: np.nanmax(x, axis=0)

agg_comm = {
    'aet': max_func,
    'pet': max_func,
    'def': sum_func,
    'pdsi': sum_func,
    'pr': sum_func,
    'soil': sum_func,
    'ro': sum_func,
    'swe': avg_func,
    'tmmn': avg_func,
    'tmmx': avg_func
}


def np_agg(tens: np.ndarray, dates: np.ndarray, att2in: dict):
    in2att = {i: a for (a, i) in zip(list(att2in.keys()), list(att2in.values()))}
    months = np.array([int(d[4:]) for d in dates])
    years = np.array([int(d[:4]) for d in dates])
    strt = np.where(months == 1)[0][0]
    stp = np.where(months == 12)[0][-1]
    months = months[strt:stp+1]
    years = np.sort(np.unique(years[strt:stp+1]))
    tens = tens[:, strt:stp+1].transpose((1, 0, 2))
    agg_tens = np.zeros((len(years), tens.shape[1], tens.shape[2]))
    for i in range(len(years)):
        for j in range(agg_tens.shape[-1]):
            agg_tens[i, :, j] = agg_comm[in2att[j]](tens[12*i: 12*(i+1), :, j])
    return agg_tens.transpose((1, 0, 2)), years


def stat_tens(df: pd.DataFrame, atts: list=('aspect', 'lat', 'lon', 'slope')):
    st = np.zeros((df.shape[0], len(atts)))
    inds = {a: i for i, a in enumerate(atts)}
    for a in atts:
        st[:, inds[a]] = df[a]
    return st, inds


def load_csv_tensor(path: str, stats: list=('aspect', 'slope')):
    df = pd.read_csv(path)
    N = df.shape[0]
    timeseries = [c for c in df.columns.values if str(c[0]).isdigit()]
    atts = np.unique([c.split('_')[-1] for c in timeseries])
    dates = np.sort(np.unique([c.split('_')[0] for c in timeseries]))
    att2in = {a: i for i, a in enumerate(atts)}
    tens = np.zeros((N, len(dates), len(atts)))
    for i, d in enumerate(dates):
        for a in atts:
            tens[:, i, att2in[a]] = df['_'.join([d, a])]
    tens, years = np_agg(tens, dates, att2in)
    ndvi = np.array([df['n' + str(y)] for y in years]).T

    st, st2in = stat_tens(df, atts=stats)
    tensor = np.concatenate([tens, np.repeat(st[:, None], tens.shape[1], axis=1)], axis=2)
    ind2feat = {i: a for (a, i) in
                zip(list(att2in.keys())+list(st2in.keys()),
                    list(att2in.values()) + list(np.array(list(st2in.values()))+max(att2in.values())+1))}
    tensor = np.concatenate([tensor, ndvi[:, :, None]], axis=2)
    ind2feat[tensor.shape[-1]-1] = 'ndvi'
    feat2ind = {a: i for (i, a) in zip(list(ind2feat.keys()), list(ind2feat.values()))}
    return tensor, feat2ind, ind2feat, years


def tensor_to_features(tensor: np.ndarray, feat2ind: dict, att: str='ndvi', lookback: int=1,
                       remove_att: list=None, feat_names: bool=True):
    assert lookback >= 1, 'Number of look-back years must be larger than 0'
    y = tensor[:, lookback:, feat2ind[att]]
    X = np.zeros((tensor.shape[0], y.shape[1], lookback, tensor.shape[-1]))
    for i in range(y.shape[1]): X[:, i] = tensor[:, i:i+lookback]
    if remove_att is not None:
        for a in remove_att: X = np.delete(X, feat2ind[a], axis=3)
    y = y.T.reshape(-1)
    # y = y.reshape(-1)
    X = X.transpose((1, 0, 2, 3)).reshape(len(y), -1)
    # X = X.reshape(len(y), -1)
    if feat_names:
        ind2feat = {i: a for (a, i) in zip(list(feat2ind.keys()), list(feat2ind.values()))}
        names = [[(ind2feat[i] + '_-{}y'.format(t+1) if lookback > 1 else ind2feat[i])
                  for i in range(tensor.shape[-1])
                  if remove_att is None or (remove_att and ind2feat[i] not in remove_att)]
                 for t in range(lookback)]

        return X, y, np.array(names).flatten()
    return X, y


if __name__ == '__main__':
    tens, f2i, _, _ = load_csv_tensor('data/test2.csv')
    X, y, features = tensor_to_features(tens, f2i, lookback=1, remove_att=['ndvi', 'aet'])
    print(features)

