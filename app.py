from flask import Flask, request, jsonify
import requests
import pandas as pd
from jaccard_method import jaccard_recommender,convert_to_set,preprocessing

app = Flask(__name__)

popular_movies = pd.read_pickle('/Users/alexangelbracho/Desktop/GitHub_projects/Movie Recommender/Movie-Recommender/popular_movies.pkl')
popular_movies_json = popular_movies.to_json()

df_movies = pd.read_csv('/Users/alexangelbracho/Desktop/GitHub_projects/Movie Recommender/Movie-Recommender/movies.csv')

df_movies = preprocessing(df_movies)

df_movies['genre_set'] = df_movies['genres'].apply(lambda x :convert_to_set(x))

@app.route('/Greetings',methods=['GET'])
def hello_world():
    return "Hi everyone"

@app.route('/Goodbye',methods=['GET'])
def goodbye_world():
    return "See you later 2"

@app.route('/popular_movies',methods=['GET'])
def most_popular_movies():
    return jsonify(popular_movies_json)

@app.route('/recommend_movies/<movieId>',methods=['GET'])
def recommend_movies(movieId):
    # recommend movies
    df_recommend_movies = jaccard_recommender(movieId,df_movies,10)
    df_recommend_movies_json = df_recommend_movies.to_json()
    return jsonify(df_recommend_movies_json)