import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

stat_feats = ['aspect', 'slope', 'lat', 'lon']

# possible aggregations (averaging, summing, minimum and maximum)
avg_func = lambda x: np.mean(x, axis=0)
sum_func = lambda x: np.sum(x, axis=0)
min_func = lambda x: np.nanmin(x, axis=0)
max_func = lambda x: np.nanmax(x, axis=0)

agg_funcs = {
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
    """
    Aggregate the features according to predefined protocols
    :param tens: the full time tensor; a numpy array with shape (N, T*M, d)
    :param dates: the dates for the time series dimension of tens
    :param att2in: a dictionary mapping from attribute name to index in the last dimension of tens
    :return: the aggregated tensor; a numpy array with shape (N, T, d)
             a list of the years used in the data
    """
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
            agg_tens[i, :, j] = agg_funcs[in2att[j]](tens[12 * i: 12 * (i + 1), :, j])
    return agg_tens.transpose((1, 0, 2)), years


def stat_tens(df: pd.DataFrame, atts: list=('aspect', 'lat', 'lon', 'slope')):
    st = np.zeros((df.shape[0], len(atts)))
    inds = {a: i for i, a in enumerate(atts)}
    for a in atts:
        st[:, inds[a]] = df[a]
    return st, inds


def load_time_tens(df: pd.DataFrame, dates: np.ndarray, att2in: dict):
    N = df.shape[0]
    atts = list(att2in.keys())
    tens = np.zeros((N, len(dates), len(atts)))
    for i, d in enumerate(dates):
        for a in atts:
            tens[:, i, att2in[a]] = df['_'.join([d, a])]
    return tens


def load_csv(path: str='data/train.csv', remove_atts: list=tuple()):
    df = pd.read_csv(path)
    timeseries = [c for c in df.columns.values if str(c[0]).isdigit()]
    atts = np.unique([c.split('_')[-1] for c in timeseries])
    atts = np.array([a for a in atts if a not in remove_atts])
    dates = np.sort(np.unique([c.split('_')[0] for c in timeseries]))
    att2in = {a: i for i, a in enumerate(atts)}
    return df, dates, att2in


def load_csv_tensor(path: str='data/train.csv', stats: list=('aspect', 'slope'), return_years: bool=False,
                    remove_atts: list=tuple(['swe'])):
    """
    Load a data tensor corresponding to the supplied CSV
    :param path: the path to the data .csv file
    :param stats: which static features to extract and add to the tensor from the csv
    :param return_years: a boolean indicating whether to return the years the data points where measured at
    :param remove_atts: a (possibly empty) list of attributes that should be dropped
    :param return_coors: a boolearn indicating whether to return the coordinates of the data points
    :return: a numpy array with shape (N, T, d) where N is the number of data points, T is the number of years
             and d is the number of features for each point
             a dictionary mapping from feature names to indices in the last dimension of the tensor
             another dictionary mapping from indices in the last dimension of the tensor to feature names
             a list of the years used to build the tensor
    """
    df, dates, att2in = load_csv(path, remove_atts)

    tens = load_time_tens(df, dates, att2in)
    tens, years = np_agg(tens, dates, att2in)
    ndvi = np.array([df['n' + str(y)] for y in years]).T

    st, st2in = stat_tens(df, atts=stats)
    tensor = np.concatenate([tens, np.repeat(st[:, None], tens.shape[1], axis=1)], axis=2)
    ind2feat = {i: a for (a, i) in
                zip(list(att2in.keys())+list(st2in.keys()),
                    list(att2in.values()) + list(np.array(list(st2in.values()))+max(att2in.values())+1))}
    tensor = np.concatenate([tensor, ndvi[:, :, None]], axis=2)
    ind2feat[tensor.shape[2]-1] = 'ndvi'
    feat2ind = {a: i for (i, a) in zip(list(ind2feat.keys()), list(ind2feat.values()))}
    if return_years: return tensor, feat2ind, ind2feat, years
    return tensor, feat2ind, ind2feat


def tensor_to_features(tensor: np.ndarray, feat2ind: dict, att: str='ndvi', lookback: int=1, feat_names: bool=True,
                       remove_att: bool=True, return_years: bool=False, difference: int=False):
    """
    Reshape the tensor into feature and output vectors (the corresponding X and y in ML)
    :param tensor: the sample-date-feature tensor created by load_csv_tensor
    :param feat2ind: a dictionary mapping from feature names to indexes
    :param att: the attribute that should be predicted by the models that will be trained
    :param lookback: the number of previous years the model can see for the prediction
    :param feat_names: a boolearn indicating whether the feature names should be returned or not
    :param remove_att: whether to remove the attribute that is to be predicted from the feature vector (as in the one
                       from a year before the prediction that should be made)
    :param return_years: a boolean indicating whether to return a list of the years of each of the samples or not
    :param difference: make predictions the difference with the previous year instead of the ndvi directly
    :return: X, a (# samples, # features) ndarray, and y, a (# samples, ) ndarray
    """
    assert lookback >= 1, 'Number of look-back years must be larger than 0'
    X, y = [], []
    tinds = [i for (f, i) in zip(list(feat2ind.keys()), list(feat2ind.values())) if f not in stat_feats and
             not (f == att and remove_att)]
    sinds = [i for (f, i) in zip(list(feat2ind.keys()), list(feat2ind.values())) if f in stat_feats]
    years = []
    for i in range(tensor.shape[1]-lookback-1):
        vecs = tensor[:, i:i+lookback, tinds].reshape((tensor.shape[0], lookback*len(tinds)))
        X.append(np.concatenate([vecs, tensor[:, i+lookback, sinds]], axis=1))
        if not difference: y.append(tensor[:, i+lookback+1, feat2ind[att]])
        else: y.append(tensor[:, i+lookback+1, feat2ind[att]] - tensor[:, i+lookback, feat2ind[att]])
        years.append(np.ones(len(y[-1]))*i)
    X, y = np.array(X), np.array(y)
    X, y = X.reshape(-1, X.shape[-1]), y.reshape(-1)
    years = np.array(years).reshape(-1)
    if feat_names:
        ind2feat = {i: a for (a, i) in zip(list(feat2ind.keys()), list(feat2ind.values()))}
        tnames = [[(ind2feat[i] + '_{}y'.format(-lookback+t) if lookback > 1 else ind2feat[i]) for i in tinds]
                  for t in range(lookback)]
        names = list(np.array(tnames).flatten()) + [ind2feat[i] for i in sinds]
        if return_years: return X, y, names, years
        return X, y, names
    if return_years: return X, y, years
    return X, y


def load_learnable(remove_att: bool=True, return_dates: bool=True, difference: bool=False):
    tens, f2i, _, years = load_csv_tensor('data/train.csv', stats=['aspect', 'slope'], return_years=True)
    return tensor_to_features(tens, f2i, lookback=1, remove_att=remove_att, return_years=return_dates,
                              difference=difference)


if __name__ == '__main__':
    # df, dates, att2in = load_csv()
    # tens = load_time_tens(df, dates, att2in)
    # tens2, years = np_agg(tens, dates, att2in)
    # print(tens.shape, tens2.shape)
    # print(len(dates), len(years))
    # print(att2in)
    t, f2i, i2f = load_csv_tensor()
    # print(f2i)
    # print(t.shape)
    # X, y, n = tensor_to_features(t, f2i, lookback=2, remove_att=True)
    # print(n)
    # print(X.shape)
    ndvi = t[:, :, f2i['ndvi']]
    w = 1
    ndvi = ndvi[:, w:] - np.array([np.mean(ndvi[:, i:i+w], axis=1) for i in range(0, ndvi.shape[1]-w)]).T
    print(ndvi.shape)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(ndvi[:10].T)
    plt.plot(np.mean(ndvi, axis=0), 'k', lw=2)
    plt.show()
