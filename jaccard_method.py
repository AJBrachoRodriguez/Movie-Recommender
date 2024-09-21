## libraries
############

# mathematical computing
import pandas as pd
import numpy as np

# sklearn
import sklearn as sk

# API
##from fastapi import FastAPI as API
##from routes import router

# visualization
import matplotlib as mpt

# global variables
global df_movies, df_ratings, df_final

## load the data
################

df_movies = pd.read_csv('/Users/alexangelbracho/Desktop/GitHub_projects/Movie Recommender/Movie-Recommender/movies.csv')
df_movies.head(5)

## data cleaning
################

def convert_to_set(string):
    list_ = list(string.split("|"))
    set_ = set(list_)
    return set_

def preprocessing(df: pd.DataFrame):
    '''It eliminates the null and duplicated values from the dataframe. Additionally,
    it transforms the columns of object type into string type.'''
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.convert_dtypes()
    df['content'] = df['genres'].str.replace('|',' ')
    #df['genre_set'] = df['genres'].str.replace('|',',')
    #df['timestamp'] = pd.to_datetime(df['timestamp'])

    return df

df_movies = preprocessing(df_movies)

df_movies['genre_set'] = df_movies['genres'].apply(lambda x :convert_to_set(x))

## function: jaccard_method
###########################

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection/union if union != 0 else 0

## function: jaccard_recommender
################################

def jaccard_recommender(movieId:int, df: pd.DataFrame, n_of_recommendations: int):
    setItem =  df.loc[df['movieId']==movieId,'genre_set'].values[0]
    #setItem = df['genre_set'].values[movieId]
    df = df[df['movieId'] != movieId]
    df['similarity'] = df.apply(lambda row: jaccard_similarity(setItem,row['genre_set']),axis=1)
    df = df.query('similarity != 0')
    df = df[['movieId','title','genres','similarity']].sort_values(by='similarity',ascending=False)
    df = df.head(n_of_recommendations)
    return df

recommendations = jaccard_recommender(4,df_movies,16)
recommendations.head()