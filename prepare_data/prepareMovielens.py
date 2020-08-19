'''
Author: Manqing Dong, 2020
'''

# ========================================================
# Data preprocessing
# MovieLens
#   userIDs range between 1 and 6040
#   movieIDs range between 1 and 3952
#       - user info:
#           gender,
#           age (1, 18, 25, 35, 45, 50, 56),
#           occupation (0 - 20)
#       - movie info:
#           title
#           year
#           genres (action, adventure, etc.)
#           director
#           rate
#       - rating info:
#           mean_rating: 3.58
# Basic info:
# start time -- 2000-04-26 09:05:32, mid_time 2000-12-03 01:52:18, end time 2003-03-01 04:49:50
# user-item -- min count: 20,  avg count: 165,  max count: 2314
# item-user -- min count: 1,  avg count: 269,  max count: 3428
# user state -- warm user: 5400, cold user: 640
# item state -- warm item: 1683, cold item: 1645

import pandas as pd
import datetime
import os
from tqdm import tqdm
from prepare_data.prepareList import *
import numpy as np


def load_movielens():
    path = 'data_raw/ml-1m/'
    user_info_path = "{}/users.dat".format(path)
    item_info_path = "{}/movies_extrainfos.dat".format(path)
    rating_path = '{}/ratings.dat'.format(path)

    user_info = pd.read_csv(user_info_path,
                            names=['user_id', 'gender', 'age', 'occupation_code', 'zip'],
                            sep="::", engine='python')
    user_info = user_info.drop(columns=['zip'])
    item_info = pd.read_csv(item_info_path,
                            names=['item_id', 'title', 'year', 'rate', 'released', 'genre', 'director', 'writer', 'actors', 'plot', 'poster'],
                            sep="::", engine='python', encoding="utf-8")
    item_info = item_info.drop(columns=['released', 'writer', 'actors', 'plot', 'poster'])
    ratings = pd.read_csv(rating_path,
                          names=['user_id', 'item_id', 'rating', 'timestamp'],
                          sep="::", engine='python')

    ratings['time'] = ratings["timestamp"].map(lambda x: datetime.datetime.fromtimestamp(x))
    ratings = ratings.drop(["timestamp"], axis=1)

    return user_info, item_info, ratings


def id_storing_movielens(max_count=20):
    storing_path = 'data_processed'
    dataset = 'movielens'

    if not os.path.exists('{}/{}/user_state_ids.p'.format(storing_path, dataset)):
        _, _, ratings_dat = load_movielens()

        sorted_time = ratings_dat.sort_values(by='time', ascending=True).reset_index(drop=True)
        start_time, split_time, end_time = sorted_time['time'][0], sorted_time['time'][
            round(0.8 * len(ratings_dat))], sorted_time['time'][len(ratings_dat) - 1]

        print('start time %s, split_time %s, end time %s' % (start_time, split_time, end_time))

        sorted_users = ratings_dat.sort_values(by=['user_id', 'time'], ascending=[True, True]).reset_index(drop=True)

        # user_statistics
        user_warm_list, user_cold_list, user_counts = [], [], []
        new_df = pd.DataFrame()

        user_ids = ratings_dat.user_id.unique()
        n_user_ids = ratings_dat.user_id.nunique()

        for u_id in tqdm(user_ids):
            u_info = sorted_users.loc[sorted_users.user_id == u_id].reset_index(drop=True)
            u_count = len(u_info)
            if u_count > max_count-1:
                new_u_info = u_info.iloc[:max_count, :]
                new_df = new_df.append(new_u_info, ignore_index=True)
                u_time = u_info['time'][0]
                if u_time < split_time:
                    user_warm_list.append(u_id)
                else:
                    user_cold_list.append(u_id)
            user_counts.append(u_count)
        print('num warm users: %d, num cold users: %d' % (len(user_warm_list), len(user_cold_list)))
        print('min count: %d, avg count: %d, max count: %d' % (min(user_counts), len(ratings_dat) / n_user_ids, max(user_counts)))

        new_all_ids = new_df.user_id.unique()

        user_state_ids = {'user_all_ids': new_all_ids, 'user_warm_ids': user_warm_list,
                          'user_cold_ids': user_cold_list}

        # item_statistics
        sorted_items = new_df.sort_values(by=['item_id', 'time'], ascending=[True, True]).reset_index(drop=True)

        item_warm_list, item_cold_list, item_counts = [], [], []

        item_ids = sorted_items.item_id.unique()
        n_item_ids = sorted_items.item_id.nunique()

        for i_id in tqdm(item_ids):
            i_info = sorted_items.loc[sorted_items.item_id == i_id].reset_index(drop=True)
            i_count = len(i_info)
            if i_count > 10 :
                item_warm_list.append(i_id)
            else:
                item_cold_list.append(i_id)
            item_counts.append(i_count)
        print('num warm items: %d,   num cold items: %d' % (len(item_warm_list), len(item_cold_list)))
        print('min count: %d,  avg count: %d,  max count: %d' % (
        min(item_counts), len(ratings_dat) / n_item_ids, max(item_counts)))

        item_state_ids = {'item_all_ids': item_ids, 'item_warm_ids': item_warm_list,
                          'item_cold_ids': item_cold_list}

        pickle.dump(new_df, open('{}/{}/ratings_sorted.p'.format(storing_path, dataset), 'wb'))
        pickle.dump(item_state_ids, open('{}/{}/item_state_ids.p'.format(storing_path, dataset), 'wb'))
        pickle.dump(user_state_ids, open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'wb'))
    else:
        print('id information is already stored.')


def dict_storing_movielens():
    storing_path = 'data_processed'
    dataset = 'movielens'

    user_state_ids = pickle.load(open('{}/{}/user_state_ids.p'.format(storing_path, dataset), 'rb'))
    item_state_ids = pickle.load(open('{}/{}/item_state_ids.p'.format(storing_path, dataset), 'rb'))

    # store user and item dict
    user_info, item_info, _ = load_movielens()

    # user
    user_all_features = []
    user_all_ids = user_state_ids['user_all_ids']

    user_dict = {}

    for u_id in tqdm(user_all_ids):
        row = user_info.loc[user_info['user_id'] == u_id]
        feature_vector = user_converting_ml(user_row=row, age_list=list_movieLens['list_age'],
                                            gender_list=list_movieLens['list_gender'],
                                            occupation_list=list_movieLens['list_occupation'])

        user_all_features.append(feature_vector)

    user_all_features = np.array(user_all_features)

    count = 0

    for u_id in tqdm(user_all_ids):
        u_info = user_all_features[count]
        user_dict[u_id] = u_info
        count += 1

    pickle.dump(user_dict, open('{}/{}/user_dict.p'.format(storing_path, dataset), 'wb'))

    # item
    item_all_features = []

    item_all_ids = item_state_ids['item_all_ids']
    item_dict = {}

    updated_i_id = []

    year_list = list(item_info.year.unique())

    for i_id in tqdm(item_all_ids):
        row = item_info.loc[item_info['item_id'] == i_id]
        if len(row) > 0:
            feature_vector = item_converting_ml(item_row=row, rate_list=list_movieLens['list_rate'],
                                                genre_list=list_movieLens['list_genre'],
                                                director_list=list_movieLens['list_director'],
                                                year_list=year_list)
            updated_i_id.append(i_id)
            item_all_features.append(feature_vector)

    item_all_features = np.array(item_all_features)

    count = 0

    for i_id in tqdm(updated_i_id):
        i_info = item_all_features[count]
        item_dict[i_id] = i_info
        count += 1

    pickle.dump(item_dict, open('{}/{}/item_dict.p'.format(storing_path, dataset), 'wb'))
