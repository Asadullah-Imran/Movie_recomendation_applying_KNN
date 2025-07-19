// DOM Elements
const heroSearch = document.getElementById("heroSearch");
const heroSearchBtn = document.getElementById("heroSearchBtn");
const recommendSearch = document.getElementById("recommendSearch");
const recommendAutocomplete = document.getElementById("recommendAutocomplete");

// Section Elements
const homeSection = getElementOrThrow("homeSection");
const searchSection = getElementOrThrow("searchSection");
const genresSection = getElementOrThrow("genresSection");
const trendingSection = getElementOrThrow("trendingSection");
const recommendationsSection = getElementOrThrow("recommendationsSection");

// Content Containers
const popularMovies = getElementOrThrow("popularMovies");
const trendingMovies = getElementOrThrow("trendingMovies");
const genreTabs = getElementOrThrow("genreTabs");
const genreMovies = getElementOrThrow("genreMovies");
const searchResults = getElementOrThrow("searchResults");
const genresContainer = getElementOrThrow("genresContainer");
const trendingContent = getElementOrThrow("trendingContent");
const selectedMovieContainer = getElementOrThrow("selectedMovie");
const recommendations = getElementOrThrow("recommendations");
const searchLoader = getElementOrThrow("searchLoader");
const recommendLoader = getElementOrThrow("recommendLoader");
const movieModal = getElementOrThrow("movieModal");
const closeModal = document.querySelector(".close-modal");
const movieDetailsContent = getElementOrThrow("movieDetailsContent");

// API Configuration
// NOTE: Update API_BASE_URL for production deployment
const API_BASE_URL = "https://movie-recomendation-applying-knnso.onrender.com"; // Updated to Render backend URL

// Helper to get element and throw if not found
function getElementOrThrow(id) {
  const el = document.getElementById(id);
  if (!el) {
    throw new Error(`Element with id '${id}' not found in DOM.`);
  }
  return el;
}

// Event Listeners for hero search
heroSearchBtn.addEventListener("click", searchMovies);
heroSearch.addEventListener("keypress", (e) => {
  if (e.key === "Enter") searchMovies();
});

// --- Recommendation Autocomplete ---
let recommendSuggestions = [];
let recommendActiveIndex = -1;

recommendSearch.addEventListener("input", async function () {
  const query = this.value.trim();
  recommendAutocomplete.innerHTML = "";
  recommendAutocomplete.classList.remove("active");
  recommendSuggestions = [];
  recommendActiveIndex = -1;
  if (query.length < 2) return;
  try {
    const response = await fetch(
      `${API_BASE_URL}/search?query=${encodeURIComponent(query)}&limit=7`
    );
    if (!response.ok) return;
    const movies = await response.json();
    if (!movies.length) return;
    recommendSuggestions = movies;
    recommendAutocomplete.innerHTML = movies
      .map(
        (m, i) =>
          `<div class="autocomplete-item" data-index="${i}">${m.clean_title} ${
            m.year ? `(${m.year})` : ""
          }</div>`
      )
      .join("");
    recommendAutocomplete.classList.add("active");
  } catch (e) {
    // Ignore errors
  }
});

recommendAutocomplete.addEventListener("mousedown", function (e) {
  if (e.target.classList.contains("autocomplete-item")) {
    const idx = parseInt(e.target.getAttribute("data-index"));
    selectRecommendSuggestion(idx);
  }
});

recommendSearch.addEventListener("keydown", function (e) {
  if (!recommendSuggestions.length) return;
  if (e.key === "ArrowDown") {
    recommendActiveIndex = Math.min(
      recommendActiveIndex + 1,
      recommendSuggestions.length - 1
    );
    updateRecommendActive();
    e.preventDefault();
  } else if (e.key === "ArrowUp") {
    recommendActiveIndex = Math.max(recommendActiveIndex - 1, 0);
    updateRecommendActive();
    e.preventDefault();
  } else if (e.key === "Enter") {
    if (recommendActiveIndex >= 0) {
      selectRecommendSuggestion(recommendActiveIndex);
      e.preventDefault();
    } else if (recommendSuggestions.length > 0) {
      selectRecommendSuggestion(0);
      e.preventDefault();
    }
  }
});

function updateRecommendActive() {
  const items = recommendAutocomplete.querySelectorAll(".autocomplete-item");
  items.forEach((item, idx) => {
    item.classList.toggle("active", idx === recommendActiveIndex);
  });
}

function selectRecommendSuggestion(idx) {
  const movie = recommendSuggestions[idx];
  if (!movie) return;
  recommendSearch.value = `${movie.clean_title} ${
    movie.year ? `(${movie.year})` : ""
  }`;
  recommendAutocomplete.innerHTML = "";
  recommendAutocomplete.classList.remove("active");
  recommendSuggestions = [];
  recommendActiveIndex = -1;
  // Trigger recommendations for this movie
  getRecommendationsForMovie(movie.movieId);
}

document.addEventListener("click", function (e) {
  if (
    !recommendAutocomplete.contains(e.target) &&
    e.target !== recommendSearch
  ) {
    recommendAutocomplete.innerHTML = "";
    recommendAutocomplete.classList.remove("active");
    recommendSuggestions = [];
    recommendActiveIndex = -1;
  }
});
// --- End Recommendation Autocomplete ---

// Initialize the app
document.getElementById("logoHome").addEventListener("click", function () {
  showSection("home");
  loadHomePage();
});
document.addEventListener("DOMContentLoaded", () => {
  showSection("home");
  loadHomePage();
  setCurrentYear();
});

// Show specific section and hide others
function showSection(section) {
  homeSection.classList.add("hidden");
  searchSection.classList.add("hidden");
  genresSection.classList.add("hidden");
  trendingSection.classList.add("hidden");
  recommendationsSection.classList.add("hidden");

  document.querySelectorAll(".nav-links a").forEach((link) => {
    link.classList.remove("active");
  });

  switch (section) {
    case "home":
      homeSection.classList.remove("hidden");
      break;
    case "search":
      searchSection.classList.remove("hidden");
      break;
    case "genres":
      genresSection.classList.remove("hidden");
      genresLink.classList.add("active");
      break;
    case "trending":
      trendingSection.classList.remove("hidden");
      trendingLink.classList.add("active");
      break;
    case "recommendations":
      recommendationsSection.classList.remove("hidden");
      break;
  }
}

// Set current year in footer
function setCurrentYear() {
  const year = new Date().getFullYear();
  document.querySelector(
    ".footer-bottom p"
  ).innerHTML = `&copy; ${year} CineMatch. All rights reserved.`;
}

// Load home page content
async function loadHomePage() {
  try {
    const response = await fetch(`${API_BASE_URL}/home`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();

    // Debug: Log the data for inspection
    console.log("Popular This Week:", data.popular_movies);
    console.log("Recently Released (Trending):", data.trending_movies);

    // Display popular movies
    displayMovies(data.popular_movies, popularMovies);

    // Display trending movies
    displayMovies(data.trending_movies, trendingMovies);

    // Setup genre tabs
    setupGenreTabs(data.top_by_genre);
  } catch (error) {
    console.error("Error loading home page:", error);
    popularMovies.innerHTML = `<div class="error"><p>Error loading popular movies: ${
      error.message || error
    }</p></div>`;
  }
}

// Load genres page content
async function loadGenresPage() {
  try {
    const response = await fetch(`${API_BASE_URL}/genres?limit=20`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const genres = await response.json();
    displayGenres(genres);
  } catch (error) {
    console.error("Error loading genres:", error);
    genresContainer.innerHTML = `<div class="error"><p>Error loading genres: ${
      error.message || error
    }</p></div>`;
  }
}

// Load trending page content
async function loadTrendingPage() {
  try {
    const response = await fetch(`${API_BASE_URL}/home`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();
    displayMovies(data.trending_movies, trendingContent);
  } catch (error) {
    console.error("Error loading trending content:", error);
    trendingContent.innerHTML = `<div class="error"><p>Error loading trending movies: ${
      error.message || error
    }</p></div>`;
  }
}

// Search movies function
async function searchMovies() {
  const query = heroSearch.value.trim();
  if (!query) return;

  showSection("search");
  searchLoader.style.display = "flex";
  searchResults.innerHTML = "";

  try {
    const response = await fetch(
      `${API_BASE_URL}/search?query=${encodeURIComponent(query)}&limit=12`
    );
    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const movies = await response.json();
    displayMovies(movies, searchResults);
  } catch (error) {
    console.error("Error searching movies:", error);
    searchResults.innerHTML = `<div class="error"><p>Error searching movies: ${
      error.message || error
    }</p></div>`;
  } finally {
    searchLoader.style.display = "none";
  }
}

// Display movies in a grid
function displayMovies(movies, container) {
  if (!movies || movies.length === 0) {
    container.innerHTML = `<div class="error"><p>No movies found</p></div>`;
    return;
  }

  container.innerHTML = movies
    .map(
      (movie) => `
        <div class="movie-card" data-id="${movie.movieId}">
            ${
              movie.poster_url
                ? `<img src="${movie.poster_url}" alt="${movie.title}" class="movie-poster">`
                : `<div class="movie-poster placeholder">
                    <i class="fas fa-film"></i>
                    <span>No Image Available</span>
                </div>`
            }
            <div class="movie-info">
                <h3 class="movie-title">${movie.clean_title}</h3>
                <div class="movie-year">${
                  movie.year ? `(${movie.year})` : ""
                }</div>
                ${
                  movie.avg_rating
                    ? `
                    <div class="movie-rating">
                        <i class="fas fa-star"></i>
                        <span>${movie.avg_rating.toFixed(1)}</span>
                        <span class="rating-count">(${
                          movie.rating_count || 0
                        })</span>
                    </div>
                `
                    : ""
                }
                <div class="movie-genres">
                    ${movie.genres
                      .split("|")
                      .map(
                        (genre) => `
                        <span class="genre-tag">${genre}</span>
                    `
                      )
                      .join("")}
                </div>
            </div>
        </div>
    `
    )
    .join("");

  // Add click event to each movie card
  container.querySelectorAll(".movie-card").forEach((card) => {
    card.addEventListener("click", () => {
      const movieId = card.getAttribute("data-id");
      showMovieDetails(movieId);
    });
  });
}

// Display genres
function displayGenres(genres) {
  if (!genres || genres.length === 0) {
    genresContainer.innerHTML = `<div class="error"><p>No genres found</p></div>`;
    return;
  }

  genresContainer.innerHTML = genres
    .map(
      (genre) => `
        <div class="genre-card" data-genre="${genre.genre}">
            <h4>${genre.genre}</h4>
            <p>${genre.count} movies available</p>
            <div class="genre-movies">
                ${genre.sample_movies
                  .slice(0, 3)
                  .map(
                    (movie) => `
                    ${
                      movie.poster_url
                        ? `<img src="${movie.poster_url}" alt="${movie.title}" class="genre-movie-poster">`
                        : `<div class="genre-movie-poster placeholder">
                            <i class="fas fa-film"></i>
                        </div>`
                    }
                `
                  )
                  .join("")}
            </div>
        </div>
    `
    )
    .join("");

  // Add click event to each genre card
  genresContainer.querySelectorAll(".genre-card").forEach((card) => {
    card.addEventListener("click", () => {
      const genre = card.getAttribute("data-genre");
      getMoviesByGenre(genre);
    });
  });
}

// Setup genre tabs
function setupGenreTabs(genreData) {
  if (!genreData) return;

  // Create tabs for first 5 genres
  const genres = Object.keys(genreData).slice(0, 5);
  genreTabs.innerHTML = genres
    .map(
      (genre, index) => `
        <div class="genre-tab ${
          index === 0 ? "active" : ""
        }" data-genre="${genre}">${genre}</div>
    `
    )
    .join("");

  // Add click event to tabs
  genreTabs.querySelectorAll(".genre-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      genreTabs
        .querySelectorAll(".genre-tab")
        .forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const genre = tab.getAttribute("data-genre");
      displayMovies(genreData[genre], genreMovies);
    });
  });

  // Show first genre by default
  if (genres.length > 0) {
    displayMovies(genreData[genres[0]], genreMovies);
  }
}

// Get movies by genre
async function getMoviesByGenre(genre) {
  showSection("search");
  searchLoader.style.display = "flex";
  searchResults.innerHTML = "";

  try {
    const response = await fetch(`${API_BASE_URL}/recommend`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        genres: [genre],
        k: 12,
      }),
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const data = await response.json();
    displayMovies(data.recommendations, searchResults);

    // Update search input to show current genre
    heroSearch.value = genre;
  } catch (error) {
    console.error("Error getting movies by genre:", error);
    searchResults.innerHTML = `<div class="error"><p>Error loading movies: ${
      error.message || error
    }</p></div>`;
  }
}

// Show movie details in modal
async function showMovieDetails(movieId) {
  try {
    const response = await fetch(`${API_BASE_URL}/movie/${movieId}`);
    if (!response.ok) throw new Error(`API error: ${response.status}`);

    const movie = await response.json();

    // Format directors and cast
    const directorsHtml =
      movie.directors && movie.directors.length > 0
        ? `<p><strong>Director:</strong> ${movie.directors.join(", ")}</p>`
        : "";

    const castHtml =
      movie.cast && movie.cast.length > 0
        ? `<div class="movie-cast">
                  <h4>Top Cast:</h4>
                  <div class="cast-grid">
                      ${movie.cast
                        .slice(0, 6)
                        .map(
                          (actor) => `
                          <div class="cast-member">
                              <div class="actor-name">${actor.name}</div>
                              <div class="character">${
                                actor.character || "Unknown"
                              }</div>
                          </div>
                      `
                        )
                        .join("")}
                  </div>
               </div>`
        : "";

    // Build modal content
    movieDetailsContent.innerHTML = `
            <div class="movie-modal-header">
                ${
                  movie.poster_url
                    ? `<img src="${movie.poster_url}" alt="${movie.title}" class="modal-poster">`
                    : `<div class="modal-poster placeholder">
                        <i class="fas fa-film"></i>
                        <span>No Image Available</span>
                    </div>`
                }
                <div class="modal-header-info">
                    <h2>${movie.clean_title} ${
      movie.year ? `(${movie.year})` : ""
    }</h2>
                    ${
                      movie.tagline
                        ? `<p class="tagline">${movie.tagline}</p>`
                        : ""
                    }
                    <div class="modal-meta">
                        ${
                          movie.runtime
                            ? `<span><i class="fas fa-clock"></i> ${movie.runtime} mins</span>`
                            : ""
                        }
                        ${
                          movie.avg_rating
                            ? `
                            <span class="rating">
                                <i class="fas fa-star"></i> ${movie.avg_rating.toFixed(
                                  1
                                )}/5
                                <span class="rating-count">(${
                                  movie.rating_count || 0
                                })</span>
                            </span>
                        `
                            : ""
                        }
                        <div class="movie-genres">
                            ${movie.genres
                              .split("|")
                              .map(
                                (genre) => `
                                <span class="genre-tag">${genre}</span>
                            `
                              )
                              .join("")}
                        </div>
                    </div>
                </div>
            </div>
            <div class="movie-modal-body">
                ${directorsHtml}
                
                ${
                  movie.description
                    ? `
                    <div class="movie-description">
                        <h4>Overview</h4>
                        <p>${movie.description}</p>
                    </div>
                `
                    : ""
                }
                
                ${
                  movie.trailer_url
                    ? `
                    <div class="trailer-container">
                        <h4>Trailer</h4>
                        <iframe 
                            width="100%" 
                            height="315" 
                            src="${movie.trailer_url.replace(
                              "watch?v=",
                              "embed/"
                            )}" 
                            frameborder="0" 
                            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                            allowfullscreen>
                        </iframe>
                    </div>
                `
                    : ""
                }
                
                ${castHtml}
                
                <div class="modal-actions">
                    <button class="btn btn-primary btn-lg" id="getRecommendationsBtn" data-id="${
                      movie.movieId
                    }">
                        <i class="fas fa-film"></i> See Recommendations
                    </button>
                </div>
            </div>
        `;

    // Show modal
    movieModal.style.display = "block";

    // Add event to recommendations button
    document
      .getElementById("getRecommendationsBtn")
      .addEventListener("click", function () {
        const movieId = this.getAttribute("data-id");
        movieModal.style.display = "none";
        getRecommendationsForMovie(movieId);
      });

    // Add event to close button
    closeModal.onclick = function () {
      movieModal.style.display = "none";
    };

    // Add event to close modal when clicking outside content
    movieModal.onclick = function (event) {
      if (event.target === movieModal) {
        movieModal.style.display = "none";
      }
    };
  } catch (error) {
    console.error("Error fetching movie details:", error);
    movieDetailsContent.innerHTML = `<div class="error"><p>Error loading movie details: ${
      error.message || error
    }</p></div>`;
  }
}

// Get recommendations for a movie
async function getRecommendationsForMovie(movieId) {
  showSection("recommendations");
  recommendLoader.style.display = "flex";
  selectedMovieContainer.innerHTML = "";
  recommendations.innerHTML = "";

  try {
    // Get selected movie details
    const movieResponse = await fetch(`${API_BASE_URL}/movie/${movieId}`);
    if (!movieResponse.ok)
      throw new Error(`API error: ${movieResponse.status}`);
    const movie = await movieResponse.json();

    // Display selected movie
    selectedMovieContainer.innerHTML = `
            <div class="selected-movie">
                ${
                  movie.poster_url
                    ? `<img src="${movie.poster_url}" alt="${movie.title}" class="selected-poster">`
                    : `<div class="selected-poster placeholder">
                        <i class="fas fa-film"></i>
                    </div>`
                }
                <div class="selected-movie-info">
                    <h3>${movie.clean_title} ${
      movie.year ? `(${movie.year})` : ""
    }</h3>
                    <div class="selected-movie-genres">
                        ${movie.genres
                          .split("|")
                          .map(
                            (genre) => `
                            <span class="genre-tag">${genre}</span>
                        `
                          )
                          .join("")}
                    </div>
                </div>
            </div>
        `;

    // Get recommendations
    const recResponse = await fetch(`${API_BASE_URL}/similar/${movieId}?k=12`);
    if (!recResponse.ok) throw new Error(`API error: ${recResponse.status}`);

    const recData = await recResponse.json();
    displayMovies(recData.recommendations, recommendations);
  } catch (error) {
    console.error("Error getting recommendations:", error);
    recommendations.innerHTML = `<div class="error"><p>Error loading recommendations: ${
      error.message || error
    }</p></div>`;
  } finally {
    recommendLoader.style.display = "none";
  }
}
