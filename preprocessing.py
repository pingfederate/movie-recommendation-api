from typing import Tuple
import pandas as pd

import pandas as pd
from typing import Tuple

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and return the movies and ratings dataframes.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the movies and ratings dataframes.
    """
    movies = pd.read_csv('./dataset/movies_metadata.csv', low_memory=False)
    ratings = pd.read_csv('./dataset/ratings.csv')
    
    return movies, ratings

def preprocess_data(movies: pd.DataFrame, ratings: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocesses the movies and ratings dataframes.

    Args:
        movies (pd.DataFrame): The movies dataframe.
        ratings (pd.DataFrame): The ratings dataframe.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the preprocessed movies and ratings dataframes.
    """
    movies = movies[['id', 'original_title', 'original_language', 'vote_count']]
    movies = movies.rename(columns={'id': 'MOVIE_ID', 'original_title': 'TITLE', 'original_language': 'LANGUAGE', 'vote_count': 'VOTE_COUNT'})
    
    ratings = ratings[['userId', 'movieId', 'rating']]
    ratings = ratings.rename(columns={'userId': 'USER_ID', 'movieId': 'MOVIE_ID', 'rating': 'RATING'})
    
    movies['MOVIE_ID'] = pd.to_numeric(movies['MOVIE_ID'], errors='coerce')
    ratings['MOVIE_ID'] = pd.to_numeric(ratings['MOVIE_ID'], errors='coerce')
    
    movies.dropna(inplace=True)
    movies_processed = movies[(movies['LANGUAGE'] == 'en') & (movies['VOTE_COUNT'] > 999)]
    
    ratings_processed = ratings
    
    return movies_processed, ratings_processed
