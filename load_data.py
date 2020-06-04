import os
import pandas as pd
# import rasterio as rio
import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")

# parameters:
ts_base_att = 'soil'  # the attribute that should be used as the "base" of the time series (meaning all the other
                      # attributes' time series should be aligned to it)

# possible aggregations (averaging, summing, minimum and maximum)
avg_func = lambda x: np.mean(x, axis=0)
sum_func = lambda x: np.sum(x, axis=0)
min_func = lambda x: np.nanmin(x, axis=0)
max_func = lambda x: np.nanmax(x, axis=0)

# dictionary of commands for aggregation of time series attributes across a year
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
    """
    A helper function used to aggregated the wanted statistics in the time series
    :param ts: a pandas series with the times that each data point was sampled at
    :param tensor: the data tensor that will be aggregated according to the specified rules
    :param agg_months: the months that the data should be aggregated over (starting from July since this was the first
                       instance of a datapoint in the provided data
    :param func: the function that should be used to aggregate the data (averaging, summing, min or max)
    :return: a numpy array containing the year-long aggregated statistics
    """
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
    """
    Creates data tensors out of the initialized DataFrame for ease of access later on in the code
    :param df: the time series DataFrame (the DataFrame that contains all temporal attributes)
    :param fdf: the constant's DataFrame (the DataFrame that contains all none temporal attributes)
    :return: 2 numpy arrays. The first will hold all of the time series information whose shape is [#atts, #years, x, y]
             The second array will hold all of the time-invariant information, whose shape is [#atts, x, y]. Finally, a
             2-tuple containing attribute names to indices for the time series array and the static array is also
             returned
    """
    # the shape of the time series (ts) tensor will be [#atts, #years, x, y]
    shape = (df.ts.unique().shape[0], df.img[0].shape[0], df.img[0].shape[1])
    attributes = df.att.unique()
    time_stamps = df.ts.unique()
    print(attributes)
    # create a dictionary that maps ts attribute names to indices in the output tensor
    ts_att2ind = {a: i for i, a in enumerate(attributes)}

    # create the ts tensor
    ts_tensor = []
    print('Creating tensor values for attribute:', flush=True, end=' ')
    for i, at in enumerate(attributes):
        print(at, flush=True, end=' ')
        tmp_ten = np.zeros(shape)
        for j, t in enumerate(time_stamps):
            tmp_ten[j] = df.loc[df['att'] == at].loc[df['ts'] == t]['img'][0]
        ts_tensor.append(agg_dict[at](df.loc[df['att'] == at]['ts'], tmp_ten))
    ts_tensor = np.array(ts_tensor)

    # the shape of the static (st) tensor will be [#atts, x, y]
    shape = (fdf.att.unique().shape[0], list(fdf['img'])[0].shape[0], list(fdf['img'])[0].shape[1])
    attributes = fdf.att.unique()
    # create a dictionary that maps ts attribute names to indices in the output tensor
    st_att2ind = {a: i for i, a in enumerate(attributes)}
    stat_tensor = np.zeros(shape)
    for i, at in enumerate(attributes):
        stat_tensor[i] = fdf.loc[fdf['att'] == at]['img'].iloc[0]
    return ts_tensor, stat_tensor, (ts_att2ind, st_att2ind)


def make_dataframes():
    """
    Read the raw data and create the relevant tensors from them
    :return: Creates the time series and static tensors by calling build_tensors on two DataFrames that are created
             from the data
    """
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

    return df, fdf


def features_labels_split(ts: np.ndarray, st: np.ndarray, att_ind: int, ts_dict: dict, st_dict: dict,
                          history: int = 1, surrounding: int = 0, ):
    """
    Reshape the data to pairs of features and labels, while mainting the temporal/spatial structure
    :param ts: the time series tensor created by build_tensors
    :param st: the static tensor created by build_tensors
    :param att_ind: index of attribute that will be predicted in the final model
    :param ts_dict: dictionary that maps between attribute's names and their indices in the ts tensor
    :param st_dict: dictionary that maps between attribute's names and their indices in the st tensor
    :param history: size of the look back window (i.e. how far back the model can see)
    :param surrounding: the radius around the point to use as features (i.e. how much spatial information the model has)
    :return: the function returns 3 objects:
                - A numpy array with shape [# data points, # features] containing all of the feature vectors in the
                  data. The data points will have the following shape:
                        <# data points> = [(<# years> - history), x - 2 x surrounding, y - 2 x surrounding)
                - A numpy array with shape [# data points,] containing all of the labels of the data
                - A list of the names of each feature, which will be used later on for feature importance. The length
                  of this list will be <# features>. The number of features each data point will have is dependent on
                  the function inputs. It's calculation is as follows:
                        <# features> = (<# ts attributes> + <# st attributes>) x <history> x (2x<surrounding> + 1)^2
                  As you can see, surrounding is the radius around the point that the model will see.
    """
    n_ts = ts.shape[0]
    outp = np.zeros((ts.shape[1] - history, ts.shape[2] - 2 * surrounding, ts.shape[3] - 2 * surrounding,
                     ts.shape[0] + st.shape[0], history, 2 * surrounding + 1, 2 * surrounding + 1))

    # create vector of labels (what the model will try to predict later on)
    if surrounding > 0:
        pred = ts[att_ind, history:, surrounding:-surrounding, surrounding:-surrounding]
    else:
        pred = ts[att_ind, history:]

    # rearrange the tensor to create the features of each data point
    for t in range(ts.shape[1] - history):
        for i in range(surrounding, ts.shape[2] - surrounding):
            for j in range(surrounding, ts.shape[3] - surrounding):
                outp[t, i - surrounding, j - surrounding, :n_ts] = ts[:, t:t + history,
                                                                   i - surrounding:i + surrounding + 1,
                                                                   j - surrounding:j + surrounding + 1]
                outp[t, i - surrounding, j - surrounding, n_ts:, :] = st[:, i - surrounding:i + surrounding + 1,
                                                                      j - surrounding:j + surrounding + 1][:, None, ...]

    # create a list of feature names so that later we will be able to extract some information from feature importance
    feat_names = np.zeros((ts.shape[0] + st.shape[0], history, 2 * surrounding + 1,
                           2 * surrounding + 1)).astype(dtype=str)
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

    # reshape the output to the correct shape
    outp = outp.reshape(list(outp.shape[:-4]) + [np.prod(outp.shape[-4:])])
    return outp, pred, feat_names


def blocked_folds(X: np.ndarray, y: np.ndarray, num_splits: int = 10, spatial_boundary: int = 10,
                  temporal_boundary: int = 1, sp_block_sz: int = 20, t_block_sz: int = 3):
    """
    Create temporal-spatial "blocks" in the data to remove auto-correlation while training, validating and testing. As
    part of this process, some data will be dropped. This is to allow some padding between different temporal-spatial
    blocks, hopefully reducing as much of the auto-correlation in the data for the training
    :param X: temporal-spatial features in a numpy array with the shape [t, x, y, # features] to split into blocks
    :param y: temporal-spatial labels in a numpy array with shape [t, x, y] to split into blocks
    :param num_splits: number of folds (for k-folds) that the data will be split into
    :param spatial_boundary: how much spatial padding should be added between different blocks
    :param temporal_boundary: how much temporal padding should be added between different blocks
    :param sp_block_sz: the maximum spatial area each block will have
    :param t_block_sz: the maximum temporal area each block will have
    :return: The data without the dropped data points as well as the indices for each fold as a numpy array
    """
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


def reshape_for_optim(X: np.ndarray, y: np.ndarray, inds: np.ndarray):
    """
    Reshape the data to match the general scheme of optimization models in sklearn, and drop all nan entries
    :param X: temporal-spatial features in a numpy array with the shape [t, x, y, # features] to split into blocks
    :param y: temporal-spatial labels in a numpy array with shape [t, x, y] to split into blocks
    :return: X reshaped to [<# samples>, <# features>] and y reshaped to [<#samples>]
    """
    X, y = X.reshape(-1, X.shape[-1]), y.flatten()
    ind = ~(np.isnan(y) | np.isnan(np.sum(X, axis=1)))
    return X[ind], y[ind], inds[ind]


if __name__ == '__main__':

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
    X, y, inds = blocked_folds(X, y, num_splits=10, spatial_boundary=10, temporal_boundary=1, sp_block_sz=20,
                               t_block_sz=3)

    # flatten into proper feature and label vectors; after this step, the data should be ready for training
    X_fl, y_fl, inds = reshape_for_optim(X, y, inds)

