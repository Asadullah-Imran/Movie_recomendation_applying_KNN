from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pydantic import BaseModel, conint
from typing import List, Optional, Dict
import os
import httpx
from dotenv import load_dotenv
from cachetools import TTLCache
import asyncio
import time
import math
from datetime import datetime, timedelta
import sys

# Load environment variables
try:
    load_dotenv()
except Exception as e:
    print("Warning: .env file not found or could not be loaded.", file=sys.stderr)

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
if not TMDB_API_KEY:
    print("Error: TMDB_API_KEY not set in environment. TMDB features will not work.", file=sys.stderr)

app = FastAPI(
    title="CineMatch - Movie Recommendation Engine",
    description="Intelligent hybrid recommendation system with rich movie discovery features",
    version="2.0.0",
    openapi_tags=[
        {
            "name": "Recommendations",
            "description": "Get personalized movie recommendations"
        },
        {
            "name": "Discovery",
            "description": "Explore movie collections and trends"
        },
        {
            "name": "Movies",
            "description": "Get movie details and metadata"
        },
        {
            "name": "User Experience",
            "description": "Features to enhance user interaction"
        }
    ]
)

# Enable CORS
# NOTE: For production, restrict allow_origins to your frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create cache for TMDB responses (max 2000 items, 24 hour TTL)
poster_cache = TTLCache(maxsize=2000, ttl=86400)
genre_cache = TTLCache(maxsize=100, ttl=86400)

# Load datasets
print("Loading datasets...")
try:
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    links = pd.read_csv("data/links.csv")
except FileNotFoundError as e:
    print(f"Error: Required data file not found: {e.filename}", file=sys.stderr)
    raise HTTPException(status_code=500, detail=f"Required data file not found: {e.filename}")
except Exception as e:
    print(f"Error loading data files: {str(e)}", file=sys.stderr)
    raise HTTPException(status_code=500, detail="Error loading data files.")

# Calculate rating statistics
print("Calculating rating statistics...")
movie_ratings = ratings.groupby('movieId').agg(
    rating_count=('rating', 'count'),
    avg_rating=('rating', 'mean')
).reset_index()

# Preprocess movie titles
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
movies['clean_title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()

# Merge datasets
print("Preparing movie details...")
movie_details = movies.merge(links, on='movieId', how='left')
movie_details = movie_details.merge(movie_ratings, on='movieId', how='left')

# Fill missing ratings
movie_details['rating_count'] = movie_details['rating_count'].fillna(0)
movie_details['avg_rating'] = movie_details['avg_rating'].fillna(0)

# Format external IDs
movie_details['imdbId'] = movie_details['imdbId'].apply(
    lambda x: f"tt{str(x).zfill(7)}" if pd.notna(x) else None
)
movie_details['tmdbId'] = movie_details['tmdbId'].apply(
    lambda x: str(int(x)) if pd.notna(x) else None
)

# Extract unique genres
all_genres = set()
for genre_list in movies['genres'].str.split('|'):
    all_genres.update(genre_list)
all_genres = sorted([g for g in all_genres if g != '(no genres listed)'])

# Create hybrid features
print("Creating hybrid features...")
valid_movies = movies['movieId'].unique()

# Collaborative filtering matrix
ratings_filtered = ratings[ratings['movieId'].isin(valid_movies)]
rating_matrix = ratings_filtered.pivot_table(
    index='movieId',
    columns='userId',
    values='rating',
    fill_value=0
).reindex(valid_movies, fill_value=0).values

# Content-based features (genres)
tfidf = TfidfVectorizer(analyzer=lambda s: s.split('|'))
content_features = tfidf.fit_transform(movies['genres']).toarray()

# Weighted hybrid features
content_weight = 0.7
collab_weight = 0.3
hybrid_features = np.hstack((
    content_weight * content_features,
    collab_weight * rating_matrix
))

# Create mappings
print("Creating mappings...")
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies['movieId'])}
idx_to_movie_id = {idx: mid for mid, idx in movie_id_to_idx.items()}
title_to_id = {title.lower().strip(): mid for title, mid in zip(movies['title'], movies['movieId'])}

# Euclidean distance calculation
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# Find nearest neighbors
def knn_find_neighbors(X_train, x_query, k):
    distances = [euclidean_distance(x_train, x_query) for x_train in X_train]
    k_indices = np.argsort(distances)[:k]
    return k_indices

# Get poster URL from TMDB with caching
async def get_poster_url(tmdb_id: str) -> str:
    """Get poster URL from TMDB API with caching"""
    if not tmdb_id or tmdb_id == "None":
        return None
    
    # Check cache first
    if tmdb_id in poster_cache:
        return poster_cache[tmdb_id]
    
    # Rate limiting protection
    await asyncio.sleep(0.1)
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {"api_key": TMDB_API_KEY}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            poster_path = data.get("poster_path")
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                # Add to cache
                poster_cache[tmdb_id] = poster_url
                return poster_url
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"TMDB ID not found: {tmdb_id}")
        else:
            print(f"TMDB API error for {tmdb_id}: {e.response.status_code}")
    except Exception as e:
        print(f"Error fetching poster for TMDB {tmdb_id}: {str(e)}")
    
    return None

# Get movie details from TMDB
async def get_movie_details_from_tmdb(tmdb_id: str) -> dict:
    """Get extended movie details from TMDB API"""
    if not tmdb_id or tmdb_id == "None":
        return None
    
    try:
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        params = {
            "api_key": TMDB_API_KEY,
            "append_to_response": "videos,credits"
        }
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
    except Exception as e:
        print(f"Error fetching TMDB details for {tmdb_id}: {str(e)}")
        return None

# Get genres from TMDB
async def get_tmdb_genres():
    """Get genre list from TMDB with caching"""
    if "genres" in genre_cache:
        return genre_cache["genres"]
    
    try:
        url = "https://api.themoviedb.org/3/genre/movie/list"
        params = {"api_key": TMDB_API_KEY}
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            genres = {g["id"]: g["name"] for g in data["genres"]}
            genre_cache["genres"] = genres
            return genres
    except Exception as e:
        print(f"Error fetching genres from TMDB: {str(e)}")
        return {}

# Pydantic models
class Movie(BaseModel):
    movieId: int
    title: str
    genres: str
    year: Optional[str] = None
    imdbId: Optional[str] = None
    tmdbId: Optional[str] = None
    clean_title: Optional[str] = None
    poster_url: Optional[str] = None
    avg_rating: Optional[float] = None
    rating_count: Optional[int] = None

class MovieDetail(Movie):
    description: Optional[str] = None
    runtime: Optional[int] = None
    tagline: Optional[str] = None
    trailer_url: Optional[str] = None
    directors: List[str] = []
    cast: List[Dict[str, str]] = []

class RecommendationRequest(BaseModel):
    title: Optional[str] = None
    movie_id: Optional[int] = None
    genres: Optional[List[str]] = None
    k: conint(ge=1, le=20) = 5

class RecommendationResponse(BaseModel):
    query_movie: Optional[str] = None
    recommendations: List[Movie]

class GenreResponse(BaseModel):
    genre: str
    count: int
    sample_movies: List[Movie]

class HomePageResponse(BaseModel):
    trending_movies: List[Movie]
    popular_movies: List[Movie]
    top_by_genre: Dict[str, List[Movie]]
    genres: List[str]

# Helper function to get movie recommendations
async def get_recommendations(movie_id: int, k: int):
    """Get recommendations for a movie ID"""
    try:
        query_idx = movie_id_to_idx[movie_id]
        
        # Get k+1 neighbors (including self)
        neighbor_indices = knn_find_neighbors(
            hybrid_features, 
            hybrid_features[query_idx], 
            k + 1
        )
        
        # Filter out self and get movie IDs
        similar_movie_ids = [
            idx_to_movie_id[i] 
            for i in neighbor_indices 
            if i != query_idx
        ][:k]
        
        # Get movie details
        recommendations = movie_details[
            movie_details['movieId'].isin(similar_movie_ids)
        ].to_dict(orient='records')
        
        # Add clean title and poster URLs
        for rec in recommendations:
            rec['clean_title'] = rec['title'].replace(f"({rec['year']})", "").strip()
            rec['poster_url'] = await get_poster_url(rec.get("tmdbId"))
        
        return recommendations
        
    except KeyError:
        raise HTTPException(status_code=404, detail="Movie not found in feature matrix")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- API Endpoints ---

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def recommend_movies(request: RecommendationRequest):
    """
    Get movie recommendations based on various criteria
    
    Parameters:
    - title: Movie title to get recommendations for
    - movie_id: Movie ID to get recommendations for
    - genres: List of genres to get recommendations for
    - k: Number of recommendations to return (1-20, default: 5)
    
    Returns:
    - JSON object with recommendations
    """
    k = request.k
    
    if request.title:
        title = request.title.strip()
        title_lower = title.lower()
        
        if title_lower not in title_to_id:
            # Try partial match
            matches = [t for t in title_to_id.keys() if title_lower in t]
            if not matches:
                raise HTTPException(status_code=404, detail="Movie not found in database")
            movie_id = title_to_id[matches[0]]
        else:
            movie_id = title_to_id[title_lower]
        
        try:
            recommendations = await get_recommendations(movie_id, k)
            return {
                "query_movie": title,
                "recommendations": recommendations
            }
        except Exception as e:
            print(f"Error in recommend_movies: {str(e)}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="An error occurred while generating recommendations. Please try again later.")
        
    elif request.movie_id:
        movie_id = request.movie_id
        if movie_id not in movie_id_to_idx:
            raise HTTPException(status_code=404, detail="Movie ID not found")
        
        try:
            recommendations = await get_recommendations(movie_id, k)
            movie_title = movie_details.loc[
                movie_details['movieId'] == movie_id, 'title'
            ].values[0]
            return {
                "query_movie": movie_title,
                "recommendations": recommendations
            }
        except Exception as e:
            print(f"Error in recommend_movies: {str(e)}", file=sys.stderr)
            raise HTTPException(status_code=500, detail="An error occurred while generating recommendations. Please try again later.")
        
    elif request.genres:
        # Get movies by genre
        genre_filter = "|".join(request.genres)
        genre_movies = movie_details[
            movie_details['genres'].str.contains(genre_filter)
        ]
        
        # Sort by popularity
        genre_movies = genre_movies.sort_values(
            ['rating_count', 'avg_rating'], 
            ascending=[False, False]
        ).head(k)
        
        # Convert to dict
        recommendations = genre_movies.to_dict(orient='records')
        
        # Add poster URLs
        for rec in recommendations:
            rec['clean_title'] = rec['title'].replace(f"({rec['year']})", "").strip()
            rec['poster_url'] = await get_poster_url(rec.get("tmdbId"))
        
        return {
            "query_movie": f"{', '.join(request.genres)} movies",
            "recommendations": recommendations
        }
        
    else:
        raise HTTPException(status_code=400, detail="No query parameters provided")

@app.get("/home", response_model=HomePageResponse, tags=["Discovery"])
async def home_page():
    """
    Get data for home page:
    - Trending movies (recently popular)
    - Popular movies (all-time favorites)
    - Top movies by genre
    - Genre list
    """
    # Get trending movies (rated in last 30 days)
    recent_cutoff = datetime.now() - timedelta(days=30)
    recent_ratings = ratings[
        pd.to_datetime(ratings['timestamp'], unit='s') > recent_cutoff
    ]
    
    if not recent_ratings.empty:
        trending = recent_ratings.groupby('movieId').agg(
            rating_count=('rating', 'count'),
            avg_rating=('rating', 'mean')
        ).reset_index()
        
        trending = trending[
            (trending['rating_count'] >= 10) & 
            (trending['avg_rating'] >= 3.5)
        ].sort_values(
            ['rating_count', 'avg_rating'], 
            ascending=[False, False]
        ).head(10)
        
        trending_movies = movie_details.merge(
            trending[['movieId']], 
            on='movieId', 
            how='inner'
        )
    else:
        trending_movies = movie_details.sort_values(
            'rating_count', ascending=False
        ).head(10)
    
    # Get popular movies (all-time)
    popular_movies = movie_details.sort_values(
        ['rating_count', 'avg_rating'], 
        ascending=[False, False]
    ).head(10)
    
    # Get top movies by genre
    top_by_genre = {}
    for genre in all_genres[:5]:  # Top 5 genres
        genre_movies = movie_details[
            movie_details['genres'].str.contains(genre)
        ].sort_values(
            ['rating_count', 'avg_rating'], 
            ascending=[False, False]
        ).head(5)
        
        # Add poster URLs
        movies_list = genre_movies.to_dict(orient='records')
        for movie in movies_list:
            movie['clean_title'] = movie['title'].replace(f"({movie['year']})", "").strip()
            movie['poster_url'] = await get_poster_url(movie.get("tmdbId"))
        
        top_by_genre[genre] = movies_list
    
    # Add poster URLs to trending and popular
    trending_list = trending_movies.to_dict(orient='records')
    for movie in trending_list:
        movie['clean_title'] = movie['title'].replace(f"({movie['year']})", "").strip()
        movie['poster_url'] = await get_poster_url(movie.get("tmdbId"))

    popular_list = popular_movies.to_dict(orient='records')
    for movie in popular_list:
        movie['clean_title'] = movie['title'].replace(f"({movie['year']})", "").strip()
        movie['poster_url'] = await get_poster_url(movie.get("tmdbId"))

    return {
        "trending_movies": trending_list,
        "popular_movies": popular_list,
        "top_by_genre": top_by_genre,
        "genres": all_genres
    }

@app.get("/genres", response_model=List[GenreResponse], tags=["Discovery"])
async def get_genres(limit: int = Query(10, ge=1, le=20)):
    """
    Get movies grouped by genre with sample movies
    
    Parameters:
    - limit: Number of genres to return
    
    Returns:
    - List of genres with sample movies
    """
    result = []
    
    for genre in all_genres[:limit]:
        genre_movies = movie_details[
            movie_details['genres'].str.contains(genre)
        ].sort_values(
            ['rating_count', 'avg_rating'], 
            ascending=[False, False]
        ).head(3)
        
        # Add poster URLs
        movies_list = genre_movies.to_dict(orient='records')
        for movie in movies_list:
            movie['clean_title'] = movie['title'].replace(f"({movie['year']})", "").strip()
            movie['poster_url'] = await get_poster_url(movie.get("tmdbId"))
        
        result.append({
            "genre": genre,
            "count": len(genre_movies),
            "sample_movies": movies_list
        })
    
    return result

@app.get("/movie/{movie_id}", response_model=MovieDetail, tags=["Movies"])
async def get_movie_details(movie_id: int):
    """
    Get details for a specific movie by ID
    
    Parameters:
    - movie_id: Movie ID to retrieve
    
    Returns:
    - Movie details with extended information
    """
    movie = movie_details[movie_details['movieId'] == movie_id]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    movie_data = movie.iloc[0].to_dict()
    movie_data['clean_title'] = movie_data['title'].replace(
        f"({movie_data['year']})", ""
    ).strip()
    movie_data['poster_url'] = await get_poster_url(movie_data.get("tmdbId"))
    
    # Get extended details from TMDB
    tmdb_data = await get_movie_details_from_tmdb(movie_data.get("tmdbId"))
    if tmdb_data:
        movie_data['description'] = tmdb_data.get('overview')
        movie_data['runtime'] = tmdb_data.get('runtime')
        movie_data['tagline'] = tmdb_data.get('tagline')
        
        # Get trailer URL
        if 'videos' in tmdb_data and 'results' in tmdb_data['videos']:
            for video in tmdb_data['videos']['results']:
                if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                    movie_data['trailer_url'] = f"https://www.youtube.com/watch?v={video['key']}"
                    break
        
        # Get directors
        movie_data['directors'] = [
            crew['name'] for crew in tmdb_data.get('credits', {}).get('crew', [])
            if crew['job'] == 'Director'
        ]
        
        # Get top cast
        movie_data['cast'] = [
            {"name": member['name'], "character": member['character']}
            for member in tmdb_data.get('credits', {}).get('cast', [])[:5]
        ]
    
    return movie_data

@app.get("/search", response_model=List[Movie], tags=["Movies"])
async def search_movies(
    query: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=50)
):
    """
    Search for movies by title
    
    Parameters:
    - query: Search string (min 2 characters)
    - limit: Maximum results to return (1-50, default: 10)
    
    Returns:
    - List of matching movies with details
    """
    # Case-insensitive search
    results = movie_details[
        movie_details['clean_title'].str.contains(query, case=False)
    ]
    
    # Prioritize exact matches at beginning of title
    results['match_score'] = results['clean_title'].apply(
        lambda t: 1 if t.lower().startswith(query.lower()) else 0
    )
    results = results.sort_values(
        ['match_score', 'year'], 
        ascending=[False, False]
    ).head(limit)
    
    # Add clean title
    results = results.assign(
        clean_title=results['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    )
    
    # Convert to list of dicts
    movie_list = results.to_dict(orient='records')
    
    # Add poster URLs asynchronously
    for movie in movie_list:
        movie['poster_url'] = await get_poster_url(movie.get("tmdbId"))
    
    return movie_list

@app.get("/similar/{movie_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_similar_movies(
    movie_id: int, 
    k: int = Query(5, ge=1, le=20)
):
    """
    Get movies similar to a specific movie
    
    Parameters:
    - movie_id: Movie ID to find similar movies for
    - k: Number of recommendations to return
    
    Returns:
    - List of similar movies
    """
    if movie_id not in movie_id_to_idx:
        raise HTTPException(status_code=404, detail="Movie ID not found")
    
    try:
        recommendations = await get_recommendations(movie_id, k)
        movie_title = movie_details.loc[
            movie_details['movieId'] == movie_id, 'title'
        ].values[0]
        return {
            "query_movie": movie_title,
            "recommendations": recommendations
        }
    except Exception as e:
        print(f"Error in get_similar_movies: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="An error occurred while generating similar recommendations. Please try again later.")

@app.get("/health", include_in_schema=False)
async def health_check():
    return {
        "status": "active",
        "movies": len(movies),
        "ratings": len(ratings),
        "poster_cache_size": len(poster_cache),
        "last_updated": pd.Timestamp.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)