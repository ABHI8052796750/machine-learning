import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import streamlit as st
import pickle

# Load the data
anime_dict = pickle.load(open('anime_name_dict.pkl', 'rb'))
anime_df = pd.DataFrame(anime_dict)

similarity = pickle.load(open('similarity.pkl', 'rb'))


# Define the recommendation function
def recommend(anime_name):
    try:
        # Find the index of the anime
        anime_index = anime_df[anime_df['name'] == anime_name].index[0]

        # Calculate the distances
        distances = similarity[anime_index]

        # Sort the distances and get top 5 recommendations
        anime_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        recommended_anime_movies = []
        for i in anime_list:
            recommended_anime_movies.append(anime_df.iloc[i[0]]['name'])
        return recommended_anime_movies
    except IndexError:
        return [f"Anime '{anime_name}' not found in the dataset."]
    except Exception as e:
        return [f"An error occurred: {e}"]


# Streamlit app
st.title('Anime Recommender System')

selected_anime_movie = st.selectbox(
    'Select an anime:',
    anime_df['name'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_anime_movie)
    for i in recommendations:
        st.write(i)
