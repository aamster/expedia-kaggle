import datetime
import time

import pandas as pd
import numpy as np


# hotel market is like NY, Boston, London
# srch-destination-id are nodes in a hierarchical taxonomy. Things like New York and vicinity,
#   New York City, JFK Airport
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split

import metrics

######################
# Trains random forest model. Uses features of prior click/book rate per user as well as
# some additional features.
# .8 MAPK CV
######################
from grid_search import GridSearch


def calc_user_num_interactions(df, click_weight=0.0, time_agg=None):
    bookings = df[df['is_booking'] == 1]
    clicks = df[df['is_booking'] == 0]

    if time_agg == 'day':
        time_agg = [df['date_time'].dt.date]
    elif time_agg == 'week':
        time_agg = [df['date_time'].dt.week, df['date_time'].dt.year]

    if time_agg:
        groupby = ['user_id', 'hotel_cluster', *time_agg]
    else:
        groupby = ['user_id', 'hotel_cluster']

    num_bookings = bookings.groupby(groupby)['is_booking'].cumsum()
    num_clicks = clicks.groupby(groupby)['cnt'].cumsum()
    df['num_bookings'] = num_bookings
    df['num_clicks'] = num_clicks

    df['num_clicks'] = df['num_clicks'].fillna(0)
    df['num_bookings'] = df['num_bookings'].fillna(0)
    df['num_interactions'] = df['num_bookings'] + click_weight * df['num_clicks']

    pivot = df.pivot_table(values='num_interactions', columns='hotel_cluster',
                        index=df.index)

    df = pd.concat([df, pivot], axis=1)

    if time_agg:
        groupby = ['user_id', *time_agg]
    else:
        groupby = ['user_id']
    df[[i for i in range(100)]] = df.groupby(groupby)[[i for i in range(100)]].shift()
    df.loc[:, [i for i in range(100)]] = df[[i for i in range(100)]].fillna(0)
    df.loc[:, [i for i in range(100)]] = df.groupby(groupby)[[i for i in range(100)]].cumsum()

    # normalize counts to 1
    row_sum = np.array(df[[i for i in range(100)]].sum(axis=1)).reshape(-1, 1)
    df.loc[:, [i for i in range(100)]] = np.divide(df[[i for i in range(100)]], row_sum)
    df.loc[:, [i for i in range(100)]] = df[[i for i in range(100)]].fillna(0)

    df = df.drop(['num_clicks', 'num_bookings', 'num_interactions'], axis=1)

    return df


def feature_engineering(X, click_weight=0.0, time_agg=None):
    X['date_time'] = pd.to_datetime(X['date_time'])
    X = calc_user_num_interactions(df=X, click_weight=click_weight, time_agg=time_agg)
    srch_ci = pd.to_datetime(X['srch_ci'])
    srch_co = pd.to_datetime(X['srch_co'])

    X['length_of_stay'] = (srch_co - srch_ci).dt.days
    X['length_of_stay'].loc[X['length_of_stay'] < 0] = -1
    X['srch_month'] = srch_ci.dt.month
    X['srch_season'] = srch_ci.apply(lambda x: (x.month % 12 + 3) // 3)
    X['month'] = X['date_time'].dt.month
    X['day_of_week'] = X['date_time'].dt.dayofweek
    X['season'] = X['date_time'].apply(lambda x: (x.month % 12 + 3) // 3)
    return X


def get_recs(y_pred, y_true):
    y_unique = np.sort(y_true.unique())
    return y_unique[(-y_pred).argsort()][:, :5]


def cross_val_performance(X, y, model, n_splits=5, shuffle=True, calc_train_performance=False):
    tss = TimeSeriesSplit(n_splits=n_splits)
    mapks = np.zeros(n_splits)
    train_mapks = np.zeros(n_splits)

    if shuffle:
        sample_index = np.arange(X.shape[0])
        np.random.shuffle(sample_index)
        X = X.iloc[sample_index]
        y = y.iloc[sample_index]

    for i, (train_index, test_index) in enumerate(tss.split(X)):
        X_train, y_train = X.iloc[train_index], y.iloc[train_index]
        X_val, y_val = X.iloc[test_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        if calc_train_performance:
            pred = model.predict_proba(X_train)
            recs = get_recs(y_pred=pred, y_true=y_train)
            mapk = metrics.map.mapk(actual=y_train, predicted=recs, k=5)
            train_mapks[i] = mapk

        pred = model.predict_proba(X_val)
        recs = get_recs(y_pred=pred, y_true=y_train)
        mapk = metrics.map.mapk(actual=y_val, predicted=recs, k=5)
        mapks[i] = mapk
    return train_mapks.mean(), mapks.mean()


def calc_cv_performance(train):
    res = []
    grid = {
        'max_depth': [None, 3],
        'max_features': ['sqrt', .4]
    }
    grid_search = GridSearch(grid=grid)

    time_aggs = [None, 'day']

    for time_agg in time_aggs:
        X_train, y_train, _, _ = preprocess_X(X=train, time_agg=time_agg)
        for param in grid_search.iter_grid():
            print(f'{datetime.datetime.now()} \tTesting {param} with {time_agg} time_agg')
            model = RandomForestClassifier(**param)
            train_mapk, val_mapk = cross_val_performance(X=X_train, y=y_train, model=model, calc_train_performance=True)
            param['train_mapk'] = train_mapk
            param['val_mapk'] = val_mapk
            param['time_agg'] = time_agg
            res.append(param)
    res = pd.DataFrame(res)
    res.sort_values('val_mapk', ascending=False).to_csv('~/Downloads/rf.csv', index=False)

    return res


def preprocess_X(X, time_agg=None, test_frac=.33):
    X = X.copy()
    X = feature_engineering(X=X, click_weight=1.0, time_agg=time_agg)
    X = X[X['is_booking'] == 1]
    drop_cols = {'date_time', 'srch_ci', 'srch_co', 'hotel_continent', 'hotel_country', 'hotel_market', 'cnt',
                 'is_booking', 'user_id'}
    X = X.sort_values('date_time')
    X = X[[c for c in X if c not in drop_cols]]
    y = X['hotel_cluster']
    X = X.drop(['hotel_cluster'], axis=1)
    X = X.fillna(0)

    train_stop = int((1-test_frac) * X.shape[0])
    X_train, y_train = X.iloc[:train_stop], y.iloc[:train_stop]
    X_test, y_test = X.iloc[train_stop:], y.iloc[train_stop:]

    return X_train, y_train, X_test, y_test


def get_test_performance(train):
    model = RandomForestClassifier()
    X_train, y_train, X_test, y_test = preprocess_X(X=train, time_agg='day')
    model.fit(X_train, y_train)
    pred = model.predict_proba(X=X_test)
    recs = get_recs(y_pred=pred, y_true=y_test)
    mapk = metrics.map.mapk(actual=y_test, predicted=recs, k=5)
    return mapk

def main():
    train = pd.read_csv('../expedia-hotel-recommendations/train.csv')

    nusers = train['user_id'].nunique()
    np.random.seed(1234)
    random_users = np.random.randint(low=train['user_id'].min(), high=train['user_id'].max(), size=int(nusers * .01))
    train = train[train['user_id'].isin(random_users)]
    # calc_cv_performance(train=train.iloc[:train_stop])

    # evaluate on test
    test_mapk = get_test_performance(train=train)
    print(test_mapk)





if __name__ == '__main__':
    main()
