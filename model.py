from typing import Tuple
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def train_model(movies_processed: pd.DataFrame, ratings_processed: pd.DataFrame) -> Tuple[NearestNeighbors, pd.DataFrame, csr_matrix]:
    """
    Trains a model for movie recommendation based on processed movie and rating data.

    Args:
        movies_processed (pd.DataFrame): Processed movie data.
        ratings_processed (pd.DataFrame): Processed rating data.

    Returns:
        Tuple[NearestNeighbors, pd.DataFrame, csr_matrix]: A tuple containing the trained model, the pivot table of movies, and the sparse matrix representation of the pivot table.
    """
    df = ratings_processed.merge(movies_processed, on='MOVIE_ID').drop_duplicates(['USER_ID', 'MOVIE_ID'])
    
    movies_pivot = df.pivot_table(columns='USER_ID', index='TITLE', values='RATING').fillna(0)
    movies_sparse = csr_matrix(movies_pivot)
    
    model = NearestNeighbors(algorithm='brute')
    model.fit(movies_sparse)
    
    return model, movies_pivot, movies_sparse
