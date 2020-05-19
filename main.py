import os
import pandas as pd
import rasterio as rio
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

# parameters:
ts_base_att = 'soil'  # the attribute that should be used as the "base" of the time series (meaning all other
# attributes time series should be aligned to it)


avg_func = lambda x: np.mean(x, axis=0)
sum_func = lambda x: np.sum(x, axis=0)
min_func = lambda x: np.nanmin(x, axis=0)
max_func = lambda x: np.nanmax(x, axis=0)

agg_dict = {
    'ndvi': (lambda x, y: agg_func(ts=x, tensor=y, agg_months=[7, 8, 9, 10], func=min_func)),
    'aet': (lambda x, y: agg_func(ts=x, tensor=y, func=max_func)),
    'pet': (lambda x, y: agg_func(ts=x, tensor=y, func=max_func)),
    'def': (lambda x, y: agg_func(ts=x, tensor=y, func=sum_func)),
    'pdsi': (lambda x, y: agg_func(ts=x, tensor=y, func=sum_func)),
    'pr': (lambda x, y: agg_func(ts=x, tensor=y, func=sum_func)),
    'soil': (lambda x, y: agg_func(ts=x, tensor=y, func=sum_func)),
    'ro': (lambda x, y: agg_func(ts=x, tensor=y, func=sum_func)),
    'tmmn': (lambda x, y: agg_func(ts=x, tensor=y, func=avg_func)),
    'tmmx': (lambda x, y: agg_func(ts=x, tensor=y, func=avg_func)),
}


def agg_func(ts: pd.Series, tensor: np.ndarray, agg_months=tuple([(7 + i) % 12 + 1 for i in range(12)]), func=avg_func):
    months = np.array([a.month for a in ts])
    if months[0] != agg_months[0]:
        s_ind = np.where(months == agg_months[0])[0][0]
        f_ind = 12 * ((months.shape[0] - s_ind) // 12) + s_ind
        months = months[s_ind:f_ind]
        tensor = tensor[s_ind:f_ind]
    indexes = np.isin(months, agg_months)
    months = months[indexes]
    tensor = tensor[indexes]
    sz = len(agg_months)
    return np.array([func(tensor[i * sz:(i + 1) * sz]) for i in range(len(months) // sz)])


def buildindex(path, printstats=False):
    # scans the files we have and returns 2 df with cols: att,ts,path. first df with the time-varying attributes and the
    #  second df with the non-time-varying attributes
    # printstats: whether to also prints some statistics
    imgpath = os.path.normpath(path)
    fnames = os.listdir(imgpath)
    imgs = [tuple(f.split('.')[0].split('_')) for f in fnames]
    df = pd.DataFrame.from_records(imgs, columns=['att', 'ts'])
    df['path'] = [os.path.join(imgpath, f) for f in fnames]  # full relative path to the image
    df.ts = pd.to_datetime(df.ts)
    df = df.sort_values(['att', 'ts'])

    fdf = df[df.ts.isnull()]  # fixed attributes / non-time-varying attributes (have NaT in the ts col)
    df = df[df.ts.notnull()]  # drop these from df
    assert len(fdf) == len(fdf.att.unique())  # make sure each of the fixed attributes has a single row
    if printstats:
        print('time varying attributes available:')
        print(df.groupby('att').size().sort_values())
        print('non-time-varying attributes available: {}'.format(fdf.att.unique()))
        # print('\nday of month breakdown:')
        # print(df.ts.dt.day.value_counts())

        # find the intersection index of all the attributes ts indexes:
        # idxint = pd.DatetimeIndex(set.intersection(*[set(ts) for att, ts in df.groupby('att').ts]))
        # intersection index of all the ts
        # tsmin, tsmax = df.ts.min(), df.ts.max()
    return df, fdf


def interpolate_ts(df):
    # temporal interpolation
    # align all attributes time-series to the base att time-series
    # interpolate any attribute that have missing values in its time-series

    # the unified time series (that all the attributes should be aligned to)
    ts = pd.DatetimeIndex(df.ts[df.att == ts_base_att]).sort_values()
    print('unified time series:', ts)

    def interpolate_att_ts(x):
        # interpolate the time series of a single attribute to aligned it to the unified ts
        if ts.equals(pd.DatetimeIndex(x.ts)):
            return x
        att = x.att.iat[0]
        assert att == 'ndvi'  # this is the only att that we expect to be unaligned
        x = x.append(pd.DataFrame({'att': att, 'ts': ts.difference(x.ts), 'path': None}),
                     ignore_index=True).sort_values('ts')
        # now x has the original ts and the unified ts with None holes we need to interpolate
        x.path = x.path.fillna(method='ffill').fillna(method='bfill')  # for now simple interpolation -
        # just take the last valid values we have (for the first one take the next valid value we have)
        x = x[x.ts.isin(ts)]
        assert x.path.notnull().all()  # make sure we have data for all the time stamps
        return x

    df = df.sort_values(['att', 'ts'])
    df = df.groupby('att').apply(interpolate_att_ts)
    return df


def build_tensors(df, fdf):
    # the shape of the output tensor will be [#atts, #years, x, y]
    shape = (df.ts.unique().shape[0], df.img[0].shape[0], df.img[0].shape[1])
    attributes = df.att.unique()
    time_stamps = df.ts.unique()
    print(attributes)
    ts_att2ind = {a: i for i, a in enumerate(attributes)}

    ts_tensor = []
    for i, at in enumerate(attributes):
        tmp_ten = np.zeros(shape)
        for j, t in enumerate(time_stamps):
            tmp_ten[j] = df.loc[df['att'] == at].loc[df['ts'] == t]['img'][0]
        ts_tensor.append(agg_dict[at](df.loc[df['att'] == at]['ts'], tmp_ten))
    ts_tensor = np.array(ts_tensor)

    shape = (fdf.att.unique().shape[0], list(fdf['img'])[0].shape[0], list(fdf['img'])[0].shape[1])
    attributes = fdf.att.unique()
    st_att2ind = {a: i for i, a in enumerate(attributes)}
    stat_tensor = np.zeros(shape)
    return ts_tensor, stat_tensor, (ts_att2ind, st_att2ind)


def build_model(df, fdf, forecast_horizon):
    """
    df: 4D data: att,time,x,y
    fdf: 3D data: att,x,y (fixed attributes (non-time-varying attributes) (have NaT in the ts col))
    forecast_horizon: forecast horizon in months

    returns:
        model
        feature importance: feature name syntax: <feature_name:att:radius:neighbor_id>
        ...

    plan:
        show hierarchy (with temporal/spatial neighbors in given radius) features importance using different models (corr, forest, ...)
            CV: use proper method without spatial/temporal autocorr
        show ranges to better describe the relations
        use pysal?
        use different lookback_win for building the features (aggregating the data in the recent time series into features)

    todo:
        drop nans

    todo optimizations:
        if we see that for different horizon the feature engineering takes long time for the same actions we can optimize it later
    """
    return None


def make_tensors():
    df, fdf = buildindex(path='data/', printstats=True)
    df = interpolate_ts(df)  # temporal interpolation

    # make sure all the tiff are aligned:
    refbounds = refres = reftrans = refcrs = refshape = None
    imgs = {}  # save the images arrays
    for i, s in df.append(fdf, ignore_index=True).drop_duplicates('path').iterrows():
        print(f'   loading image: {s.path}')
        with rio.open(s.path) as d:
            if refbounds is None:
                refbounds = d.bounds
                refres = d.res
                reftrans = d.transform
                refcrs = d.crs
                refshape = d.shape
            assert (d.count == 1) and (d.crs == refcrs) and (d.transform == reftrans) and \
                   (d.bounds == refbounds) and (d.res == refres) and (d.shape == refshape)
            img = d.read(1)
            imgs[s.path] = img

    # save the image arrays in our data structure:
    df['img'] = df.path.map(imgs)
    fdf['img'] = fdf.path.map(imgs)

    return build_tensors(df, fdf)


def ready_data(ts: np.ndarray, st: np.ndarray, att_ind: int, ts_dict: dict, st_dict: dict,
               history: int = 1, surrounding: int = 0, ):
    n_ts = ts.shape[0]
    outp = np.zeros((ts.shape[1] - history, ts.shape[2] - 2 * surrounding, ts.shape[3] - 2 * surrounding,
                     ts.shape[0] + st.shape[0], history, 2 * surrounding + 1, 2 * surrounding + 1))

    pred = ts[att_ind, history:, surrounding:-surrounding, surrounding:-surrounding]
    for t in range(ts.shape[1] - history):
        for i in range(surrounding, ts.shape[2] - surrounding):
            for j in range(surrounding, ts.shape[3] - surrounding):
                outp[t, i - surrounding, j - surrounding, :n_ts] = ts[:, t:t + history,
                                                                   i - surrounding:i + surrounding + 1,
                                                                   j - surrounding:j + surrounding + 1]
                outp[t, i - surrounding, j - surrounding, n_ts:, :] = st[:, i - surrounding:i + surrounding + 1,
                                                                      j - surrounding:j + surrounding + 1][:, None, ...]

    feat_names = np.zeros((ts.shape[0] + st.shape[0], history, 2 * surrounding + 1, 2 * surrounding + 1)).astype(
        dtype=str)
    for k in ts_dict:
        feat_names[ts_dict[k]] = k + '_'
    for k in st_dict:
        feat_names[n_ts + st_dict[k]] = k + '_'
    prefix = list(feat_names.flatten())
    for t in range(feat_names.shape[1]):
        feat_names[:, t] = 't-{}_'.format(t + 1)
    times = list(feat_names.flatten())
    for i in range(feat_names.shape[2]):
        feat_names[:, :, i] = 'X{}_'.format(i - surrounding)
    x = list(feat_names.flatten())
    for i in range(feat_names.shape[3]):
        feat_names[:, :, :, i] = 'Y{}'.format(i - surrounding)
    y = list(feat_names.flatten())
    feat_names = [prefix[i] + times[i] + x[i] + y[i] for i in range(len(prefix))]
    outp = outp.reshape(list(outp.shape[:-4]) + [np.prod(outp.shape[-4:])])
    return outp, pred, feat_names


def blocked_folds(X: np.ndarray, y: np.ndarray, num_splits: int = 10, spatial_boundary: int = 10,
                  temporal_boundary: int = 1,
                  sp_block_sz: int = 20, t_block_sz: int = 3):
    inds = -1 * np.ones(y.shape)
    counter = 0
    for t in range(0, inds.shape[0] - t_block_sz, t_block_sz + temporal_boundary):
        for i in range(0, inds.shape[1] - sp_block_sz, sp_block_sz + spatial_boundary):
            for j in range(0, inds.shape[2] - sp_block_sz, sp_block_sz + spatial_boundary):
                inds[t:t + t_block_sz, i:i + sp_block_sz, j:j + sp_block_sz] = counter
                counter += 1
    y_fold = y[inds >= 0]
    X_fold = X[inds >= 0]
    inds = inds[inds >= 0]
    inds = np.floor((num_splits - 1) * (inds / np.max(inds))).astype(int)
    return X_fold, y_fold, inds


def make_save():
    ts, st, dicts = make_tensors()
    np.save('timeseries_tensor.npy', ts)
    np.save('static_tensor.npy', st)
    with open('att_dicts.pkl', 'wb') as f: pickle.dump(dicts, f)


ts = np.load('timeseries_tensor.npy')
st = np.load('static_tensor.npy')
with open('att_dicts.pkl', 'rb') as f: ts_dict, st_dict = pickle.load(f)
X, y, names = ready_data(ts, st, ts_dict['ndvi'], ts_dict, st_dict, surrounding=1)
blocked_folds(X, y)
