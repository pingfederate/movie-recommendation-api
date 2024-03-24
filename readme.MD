# Movie Recommender API

This is a simple API for movie recommendation based on user ratings. It provides endpoints to get movie recommendations and retrieve all available movie names.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/movie-recommender-api.git
    ```

2. Navigate to the project directory:

    ```bash
    cd movie-recommender-api
    ```

3. Install dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```

## Running the API

Run the following command to start the FastAPI server:

    ```bash    
    uvicorn api:app --reload
    ```

The API will be available at http://localhost:8000.

## API Endpoints

### Get Movie Recommendations

- *URL:* `/recommendations/{movie_name}`
- *Method:* GET
- *Description:* Get a movie recommendation based on the provided movie name.
- *Parameters:*
    - `movie_name` (str): The name of the movie for which to get a recommendation.
- *Response:* `movie_recommendation` (str):  The recommended movie name.

### Get All Movie Names

- *URL:* `/movies`
- *Method:* GET
- *Description:* Get all movie names available in the dataset.
- *Response:* List of movie names (List[str]).

## Usage Example

To get a recommendation for the movie "Toy Story", you can send a GET request to the `/recommendations/Toy Story` endpoint:

    ```bash
    curl http://localhost:8000/recommendations/Tiy%20Story
    ```