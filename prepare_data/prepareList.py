'''
Author: Manqing Dong, 2020
'''

import re
import pickle


def load_list(f_name):
    list_ = []
    with open(f_name, encoding="utf-8") as f:
        for line in f.readlines():
            list_.append(line.strip())
    return list_

def pickle_load(file_name):
    path = 'data_raw/book_crossing/'
    target_file = pickle.load(open(path+file_name, 'rb'))
    return target_file


# movieLens
list_movieLens = {
    'list_age': [1, 18, 25, 35, 45, 50, 56],
    'list_gender': ['M', 'F'],
    'list_occupation': list(range(0, 21)),
    'list_genre': load_list('data_raw/ml-1m/List_genre.txt'),
    'list_rate': ['PG-13', 'UNRATED', 'NC-17', 'PG', 'G', 'R'],
    'list_director': load_list('data_raw/ml-1m/List_director.txt')
}


def user_converting_ml(user_row, age_list, gender_list, occupation_list):
    # gender_dim: 2, age_dim: 7, occupation: 21
    gender_idx = gender_list.index(user_row.iat[0, 1])
    age_idx = age_list.index(user_row.iat[0, 2])
    occupation_idx = occupation_list.index(user_row.iat[0, 3])
    return [gender_idx, age_idx, occupation_idx]


def item_converting_ml(item_row, rate_list, genre_list, director_list, year_list):
    # rate_dim: 6, year_dim: 1,  genre_dim:25, director_dim: 2186,
    rate_idx = rate_list.index(item_row.iat[0, 3])
    genre_idx = [0] * 25
    for genre in str(item_row.iat[0, 4]).split(", "):
        idx = genre_list.index(genre)
        genre_idx[idx] = 1
    director_idx = [0] * 2186
    for director in str(item_row.iat[0, 5]).split(", "):
        idx = director_list.index(re.sub(r'\([^()]*\)', '', director))
        director_idx[idx] = 1
    year_idx = year_list.index(item_row.iat[0, 2])
    out_list = list([rate_idx, year_idx])
    out_list.extend(genre_idx)
    out_list.extend(director_idx)
    return out_list


def user_converting_bk(user_row, age_list, location_list):
    age_idx = age_list.index(user_row.iat[0, 2])
    location_idx = location_list.index(user_row.iat[0, 1])
    return [age_idx, location_idx]


def item_converting_bk(item_row, author_list, year_list, publisher_list):
    author_idx = author_list.index(item_row.iat[0, 2])
    year_idx = year_list.index(item_row.iat[0, 3])
    publisher_idx = publisher_list.index(item_row.iat[0, 4])
    return [author_idx, year_idx, publisher_idx]
