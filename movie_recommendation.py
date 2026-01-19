import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

path = r"C:\Users\KIIT\OneDrive\Desktop\projects"
ratings = pd.read_csv(os.path.join(path, "ratings.csv"))
movies = pd.read_csv(os.path.join(path, "movies.csv"))
data = pd.merge(ratings, movies[['movieId', 'title']], on='movieId')

user_movie_matrix = data.pivot_table(
    index='userId',
    columns='title',
    values='rating'
).fillna(0) 

user_movie_matrix.head()

user_similarity = cosine_similarity(user_movie_matrix)

user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)



def recommend_movies(user_id, n_recommendations=10, min_ratings=4):
    if user_id not in user_movie_matrix.index:
        return "User not found."
    
    user_ratings = user_movie_matrix.loc[user_id]
    rated_movies = user_ratings[user_ratings > 0].index
    
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:11]
    
    predictions = {}
    for movie in user_movie_matrix.columns:
        if movie not in rated_movies:
            weighted_sum = 0
            sim_sum = 0
            for sim_user, similarity in similar_users.items():
                if movie in user_movie_matrix.loc[sim_user][user_movie_matrix.loc[sim_user] > 0].index:
                    rating = user_movie_matrix.loc[sim_user, movie]
                    weighted_sum += rating * similarity
                    sim_sum += abs(similarity)
            
            if sim_sum > 0:
                predictions[movie] = weighted_sum / sim_sum
    
    movie_stats = data.groupby('title')['rating'].agg(['count', 'mean']).round(2)
    filtered_preds = {m: pred for m, pred in predictions.items() if movie_stats.loc[m, 'count'] > min_ratings}

    top_movies = sorted(filtered_preds.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]
    return pd.DataFrame(top_movies, columns=['title', 'predicted_rating'])

recommendations = recommend_movies(1)
print("recommendation for 1st userid : ")
print(recommendations)
