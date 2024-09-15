import os
import pickle
import streamlit as st
import requests
import numpy as np
from scipy import sparse


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=dbd0caffd6c531ccc437c7e090f5c1a9&language=en-US".format(
        movie_id)
    try:
        data = requests.get(url)
        data.raise_for_status()  # Check for HTTP errors
        data = data.json()
        poster_path = data.get('poster_path')
        if poster_path:
            full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
            return full_path
        else:
            return "Poster not available"
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch poster: {e}")
        return "Error fetching poster"


def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index].toarray()[0])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:6]:
        # fetch the movie poster
        movie_id = movies.iloc[i[0]].movie_id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters


# Get the absolute path to the directory where the script is located
base_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the model files
movie_list_path = os.path.join(base_dir, 'venv', 'model', 'movie_list.pkl')
similarity_chunk_0_path = os.path.join(base_dir, 'venv', 'model', 'similarity_chunk_0.npz')

# Check if the files exist
if not os.path.exists(movie_list_path) or not os.path.exists(similarity_chunk_0_path):
    st.error("Model files not found. Please check the file paths.")
else:
    # Load the movie list
    movies = pickle.load(open(movie_list_path, 'rb'))

    # Load the similarity matrix chunks and combine them
    similarity_chunk_0 = sparse.load_npz(similarity_chunk_0_path)
    # Add other chunks as needed and combine them
    similarity = sparse.vstack([similarity_chunk_0])

    st.header('Movie Recommender System')

    movie_list = movies['title'].values
    selected_movie = st.selectbox(
        "Type or select a movie from the dropdown",
        movie_list
    )

    if st.button('Show Recommendation'):
        recommended_movie_names, recommended_movie_posters = recommend(selected_movie)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_movie_names[0])
            st.image(recommended_movie_posters[0])
        with col2:
            st.text(recommended_movie_names[1])
            st.image(recommended_movie_posters[1])
        with col3:
            st.text(recommended_movie_names[2])
            st.image(recommended_movie_posters[2])
        with col4:
            st.text(recommended_movie_names[3])
            st.image(recommended_movie_posters[3])
        with col5:
            st.text(recommended_movie_names[4])
            st.image(recommended_movie_posters[4])

# streamlit run file.py
