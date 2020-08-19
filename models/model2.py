import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.model_selection import KFold
import metrics


######################
# Uses collaborative filtering by bookings/clicks for existing users
######################
from grid_search import GridSearch


def get_user_hotel_preference_matrix(X, click_weight=0.0):
    user_bookings = X.groupby(['user_id', 'hotel_cluster'])['is_booking'].sum()
    user_clicks = X.groupby(['user_id', 'hotel_cluster'])['cnt'].sum() - user_bookings
    user_hotel_preference = user_bookings + click_weight * user_clicks
    user_hotel_preference = user_hotel_preference.reset_index().rename(columns={0: 'preference'})
    user_hotel_preference = user_hotel_preference.pivot_table(values='preference', index='user_id',
                                                              columns='hotel_cluster', fill_value=0)
    return user_hotel_preference


def fit(X, click_weight=0.0, n_components=10, alpha=0.0, l1_ratio=0.0, max_iter=200):
    user_hotel_preference = get_user_hotel_preference_matrix(X=X, click_weight=click_weight)
    model = NMF(n_components=n_components, init='nndsvd', random_state=0, alpha=alpha, l1_ratio=l1_ratio,
                max_iter=max_iter)
    W = model.fit_transform(X=user_hotel_preference)
    H = model.components_
    return W, H, user_hotel_preference.index


def predict(W, H, train_user_ids, X_test):
    preds = W.dot(H)
    preds = (-preds).argsort()
    preds = preds[:, :5]
    preds = pd.Series(preds.tolist(), index=train_user_ids)

    existing_user_ids = set(X_test['user_id']).intersection(train_user_ids)
    new_user_ids = set(X_test['user_id']).difference(train_user_ids)

    existing_user_preds = X_test[X_test['user_id'].isin(existing_user_ids)]['user_id'].map(preds)

    recs_new_idx = X_test[X_test['user_id'].isin(new_user_ids)].index
    most_popular_hotels = (-H.mean(axis=0)).argsort()[:5].tolist()
    new_user_preds = pd.Series([most_popular_hotels for _ in range(len(recs_new_idx))],
                               index=recs_new_idx)

    preds = pd.concat([existing_user_preds, new_user_preds])

    return preds.loc[X_test.index]


def cross_val_performance(X, n_splits=5, click_weight=0.0, n_components=10):
    kf = KFold(n_splits=n_splits, shuffle=True)
    mapks = np.zeros(n_splits)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index]
        X_val = X.iloc[test_index]

        # Only predicting when there is a booking
        # If not present, assumed to be a booking
        if 'is_booking' in X_val:
            X_val = X_val[X_val['is_booking'] == 1]

        W, H, train_user_ids = fit(X=X_train, click_weight=click_weight, n_components=n_components)
        recs = predict(W=W, H=H, train_user_ids=train_user_ids, X_test=X_val)
        mapk = metrics.map.mapk(actual=X_val['hotel_cluster'], predicted=recs.values, k=5)
        mapks[i] = mapk
    return mapks.mean()


def calc_performance(X):
    res = []
    grid = {
        'click_weight': [0.0, 0.01, 0.1, 1.0],
        'n_components': [10, 30, 50]
    }
    grid_search = GridSearch(grid=grid)

    for param in grid_search.iter_grid():
        mapk = cross_val_performance(X=X, **param)
        param['mapk'] = mapk
        res.append(param)
    pd.DataFrame(res).sort_values('mapk', ascending=False).to_csv('~/Downloads/collab-filtering.csv', index=False)


def main():
    train = pd.read_csv('../expedia-hotel-recommendations/train.csv', usecols=['hotel_cluster', 'is_booking',
                                                                               'user_id', 'cnt'])
    nusers = train['user_id'].nunique()
    np.random.seed(1234)
    random_users = np.random.randint(low=train['user_id'].min(), high=train['user_id'].max(), size=int(nusers * .01))
    train = train[train['user_id'].isin(random_users)]
    calc_performance(X=train)


if __name__ == '__main__':
    main()
