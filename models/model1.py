import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import metrics
from plotly import express as px

######################
# Uses most popular hotel by bookings/clicks for existing search destinations
# Uses most popular hotel by bookings/clicks overall for new search destinations
# .32 MAP CV
######################


def calc_hotel_popularity(df, click_weight=0.0):
    num_bookings = df.groupby(['srch_destination_id', 'hotel_cluster'])['is_booking'].sum()
    num_clicks = df.groupby(['srch_destination_id', 'hotel_cluster']).size() - num_bookings
    hotel_popularity = num_bookings + click_weight * num_clicks
    hotel_popularity = hotel_popularity.reset_index().rename(columns={0: 'hotel_popularity'})
    return hotel_popularity


def predict_for_existing_destination(X):
    X = X.sort_values('hotel_popularity', ascending=False)
    recs = X.groupby(['srch_destination_id'])['hotel_cluster'].apply(lambda x: x[:5].tolist())
    return recs


def predict_for_new_destination(X):
    hotel_popularity = X.groupby(['hotel_cluster'])['hotel_popularity'].sum()
    hotel_popularity = hotel_popularity.sort_values(ascending=False)
    recs = hotel_popularity.iloc[:5].index.tolist()
    return recs


def predict(X_train, X_test, click_weight=0.0):
    common_srch_destination = set(X_test['srch_destination_id']).intersection(X_train['srch_destination_id'])
    new_srch_destination = set(X_test['srch_destination_id']).difference(X_train['srch_destination_id'])

    X_train = calc_hotel_popularity(df=X_train, click_weight=click_weight)
    recs_common = predict_for_existing_destination(X=X_train)
    recs_new = predict_for_new_destination(X=X_train)

    recs_common = X_test[X_test['srch_destination_id'].isin(common_srch_destination)]['srch_destination_id'].map(recs_common)

    recs_new_idx = X_test[X_test['srch_destination_id'].isin(new_srch_destination)].index
    recs_new = pd.Series([recs_new for _ in range(len(recs_new_idx))],
                         index=recs_new_idx)

    recs = pd.concat([recs_common, recs_new])

    return recs.loc[X_test.index]


def cross_val_performance(X, n_splits=5, click_weight=0.0):
    kf = KFold(n_splits=n_splits)
    mapks = np.zeros(n_splits)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train = X.iloc[train_index]
        X_val = X.iloc[test_index]

        # Only predicting when there is a booking
        # If not present, assumed to be a booking
        if 'is_booking' in X_val:
            X_val = X_val[X_val['is_booking'] == 1]

        recs = predict(X_train=X_train, X_test=X_val, click_weight=click_weight)
        mapk = metrics.map.mapk(actual=X_val['hotel_cluster'], predicted=recs, k=5)
        mapks[i] = mapk
    return mapks.mean()


def plot_click_weight_performance(X):
    click_weights = [0.0, 0.01, 0.1, 1.0]
    mapks = []
    for click_weight in click_weights:
        mapk = cross_val_performance(X=X, click_weight=click_weight)
        mapks.append(mapk)
    df = pd.DataFrame({'click_weight': click_weights, 'mapk': mapks})
    fig = px.line(df, x='click_weight', y='mapk', log_x=True, title='Click-weight Performance')
    fig.update_traces(mode='markers+lines')
    fig.show()


def main():
    train = pd.read_csv('../expedia-hotel-recommendations/train.csv', usecols=['hotel_cluster', 'is_booking',
                                                                               'srch_destination_id'])
    plot_click_weight_performance(X=train)


if __name__ == '__main__':
    main()
