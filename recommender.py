# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:22:28 2024

@author: Sj
"""

import numpy as np
import pandas as pd
import ast
import pickle
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on="title")

movies['original_language'].value_counts()

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.isnull().sum()
movies.dropna(inplace=True)


# print(movies.iloc[0].genres)

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


def convertcount3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convertcount3)
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

# movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies['overview'] = movies['overview'].apply(lambda x: x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features=5000, stop_words='english')

vector = cv.fit_transform(new['tags']).toarray()

similarity = cosine_similarity(vector)

new[new['title'] == 'The Lego Movie'].index[0]


def recommend(movie):
    index = new[new['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)


recommend('Gandhi')

pickle.dump(new, open('venv/model/movie_list.pkl', 'wb'))
pickle.dump(similarity, open('venv/model/similarity.pkl', 'wb'))