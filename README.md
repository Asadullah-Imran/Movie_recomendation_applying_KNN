# CineMatch - Intelligent Movie Recommendation System

CineMatch is a full-stack web application that provides intelligent movie recommendations using a hybrid approach, including a manual implementation of the K-Nearest Neighbors (KNN) algorithm. The system combines collaborative filtering and content-based filtering to deliver personalized and relevant movie suggestions.

## Features

- **Hybrid Recommendation Engine:**
  - Uses both content-based (genre, metadata) and collaborative (user ratings) features.
  - Manual KNN implementation for finding similar movies.
- **Modern Web UI:**
  - Responsive, visually appealing interface built with HTML, CSS, and JavaScript.
  - Search, trending, genre browsing, and detailed movie modals.
- **Rich Movie Metadata:**
  - Integrates with TMDB API for posters, trailers, and cast info.
- **FastAPI Backend:**
  - RESTful API endpoints for recommendations, search, genres, and movie details.
- **Autocomplete & Modal UI:**
  - Autocomplete for movie search and recommendations.
  - Modal for detailed movie info and recommendations.

## Manual KNN Model

CineMatch implements a custom K-Nearest Neighbors (KNN) algorithm for movie recommendations:

- **Feature Construction:**
  - Content-based features: TF-IDF vectors of movie genres.
  - Collaborative features: User rating matrix (pivoted by movie and user).
  - Hybrid features: Weighted combination of content and collaborative features.
- **Distance Calculation:**
  - Uses Euclidean distance to measure similarity between movies in the hybrid feature space.
- **Neighbor Search:**
  - For a given movie, finds the `k` nearest movies (lowest distance) as recommendations.
- **No External ML Libraries for KNN:**
  - The KNN logic is implemented manually using NumPy for distance calculations and array operations.

## Backend (FastAPI)

- **Location:** `app.py`
- **Key Endpoints:**
  - `/recommend` - Get recommendations by movie, title, or genre.
  - `/similar/{movie_id}` - Get similar movies using KNN.
  - `/movie/{movie_id}` - Get detailed info for a movie.
  - `/search` - Search movies by title.
  - `/genres` - List genres and sample movies.
  - `/home` - Get home page data (popular, trending, by genre).
- **Data:**
  - Uses CSV files in `data/` for movies, ratings, links, and tags.
  - Integrates with TMDB for posters, trailers, and cast.
- **Caching:**
  - Uses in-memory cache for TMDB API responses.

## Frontend

- **Location:** `index.html`, `script.js`, `style.css`
- **Technologies:**
  - HTML5, CSS3 (custom theme, responsive), Vanilla JavaScript
- **Key Features:**
  - Search bar and autocomplete for movies.
  - Genre tabs and trending sections.
  - Movie cards with poster, title, year, rating, and genres.
  - Modal for detailed movie info and recommendations.
  - Responsive design for desktop and mobile.

## Setup & Usage

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd movie_recomendation/project2
   ```
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up TMDB API Key:**
   - Create a `.env` file and add your TMDB API key:
     ```
     TMDB_API_KEY=your_tmdb_api_key_here
     ```
4. **Run the backend server:**
   ```bash
   uvicorn app:app --reload
   ```
5. **Open `index.html` in your browser** (or serve with a static file server).

## Project Structure

```
project2/
├── app.py              # FastAPI backend with KNN logic
├── data/               # Movie, ratings, links, tags CSVs
├── index.html          # Main HTML UI
├── script.js           # Frontend logic (fetch, UI, modals)
├── style.css           # Custom styles and responsive design
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
```

## Acknowledgements

- [TMDB API](https://www.themoviedb.org/documentation/api) for movie metadata
- [FastAPI](https://fastapi.tiangolo.com/) for backend
- [scikit-learn](https://scikit-learn.org/) for TF-IDF vectorization
- [Font Awesome](https://fontawesome.com/) for icons

## License

MIT License
