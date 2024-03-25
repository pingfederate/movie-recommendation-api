from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from preprocessing import load_data, preprocess_data
from model import train_model
from random import choice

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

movies, ratings = load_data()
movies_processed, ratings_processed = preprocess_data(movies, ratings)
model, movies_pivot, movies_sparse = train_model(movies_processed, ratings_processed)

@app.get("/recommendations/{movie_name}")
async def get_recommendations(movie_name: str) -> dict:
    """
    Get movie recommendations based on a given movie name.

    Args:
        movie_name (str): The name of the movie to get recommendations for.

    Returns:
        dict: A dictionary containing the recommended movie.

    """
    _, suggestions_id = model.kneighbors(movies_pivot.loc[movie_name].values.reshape(1, -1))
    movie_list = [movie for movie in movies_pivot.index[suggestions_id[0]] if movie != movie_name]
    recommendation = choice(movie_list)
    
    return {"movie_recommendation": recommendation}

@app.get("/movies", response_model=List[str], tags=["movies"])
async def get_all_movies() -> List[str]:
    """
    Get all movie names.

    Returns:
        List[str]: A list containing all movie names.

    """
    return movies_processed["TITLE"].values.tolist()