import numpy as np
import random
from tqdm import tqdm

def data_loader_lastfm():
    path = './data/last-fm/'
    num_users, num_items = 10000, 5706
   
    user_gender = load_gender(path)
    artist_genre = load_genre(path)

    num, idx, data, gender, genre = load_data(path, 
                                              num_users, 
                                              num_items, 
                                              0.9,
                                              user_gender,
                                              artist_genre)
    
    item= {}
    item['M'] = genre['Hip-Hop']
    item['F'] = genre['indie']
    np.random.shuffle(gender['M'])
    np.random.shuffle(gender['F'])
    new_users = gender['M'][:5000] + gender['F'][:5000]
    new_users.sort()

    ngender = {'M':[], 'F':[]}
    
    for i, u in enumerate(new_users):
        if u in gender['M']:
            ngender['M'].append(i)
        else:
            ngender['F'].append(i)

    data = (data[0][new_users, :], data[1][new_users, :])
    
    return data, ngender, item

def load_gender(path):
    f = open(path + "usersha1-profile.tsv")
    lines = f.readlines()
    users = {}
    for line in lines:
        user, gender, *args = line.split("\t")
        if gender not in ['m', 'f']:
            continue
        users[user] = gender

    return users

def load_genre(path):
    f = open(path + "artist.txt")
    lines = f.readlines()
    artists= {}
    for line in lines:
        artist, *args = line.split(", ")
        artists[artist] = []
        for g in args:
            if '\n' in g:
                g = g[:-1]
            artists[artist].append(g)

    return artists

def load_data(path, num_users, num_items, train_ratio, user_gender, artist_genre):
    '''
    1. build users/artists index mapping.
    2. build rating matrix: n by m.
    '''
    f = open(path + "usersha1-artmbid-artname-plays.tsv")
    lines = f.readlines()
    random.shuffle(lines)
    users, items, dic_rating = set(), set(), {}
    train_ratio = 0.9

    for line in tqdm(lines):
        user, _, item, rating = line.split("\t")
        rating = int(rating.split("\n")[0])

        if user not in user_gender.keys():
            continue

        if item not in artist_genre.keys():
            continue

        if rating > 254:
            dic_rating[(user, item)] = 1
            users.add(user)
            items.add(item)
        else:
            dic_rating[(user, item)] = -1
            users.add(user)
            items.add(item)

    user_idx, item_idx = {}, {}
    for i, u in enumerate(users):
        user_idx[u] = i

    for i, a in enumerate(items):
        item_idx[a] = i

    num_users, num_items = len(users), len(items)
    X_train = np.zeros((num_users, num_items))
    X_test = np.zeros((num_users, num_items))

    num_ratings = len(dic_rating)

    for i, (key, value) in tqdm(enumerate(dic_rating.items())):
        if i < int(num_ratings * train_ratio):
            X_train[user_idx[key[0]], item_idx[key[1]]] = value
        else:
            X_test[user_idx[key[0]], item_idx[key[1]]] = value

    gender, genre = {'M':[], 'F':[]}, {'Hip-Hop':[], 'indie':[]}

    for i, g in tqdm(user_gender.items()):
        if i not in users: continue

        if g == 'm':
            gender['M'].append(user_idx[i])
        if g == 'f':
            gender['F'].append(user_idx[i])

    for i, g in tqdm(artist_genre.items()):
        if i not in items: continue

        if 'Hip-Hop' in g:
            genre['Hip-Hop'].append(item_idx[i])
        if 'musical' in g:
            genre['indie'].append(item_idx[i])

    return (num_users, num_items), (user_idx, item_idx), (X_train, X_test), gender, genre