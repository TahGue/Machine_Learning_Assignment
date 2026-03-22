# Movie Recommendation System

A movie recommendation system built with the MovieLens dataset, using content-based filtering, collaborative filtering, and a hybrid approach. The recommended approach for this assignment is the hybrid model, since it combines metadata similarity with rating-based signals while remaining practical to run locally.

## Features

- **Content-Based Filtering**: Uses TF-IDF vectorization on movie genres, tags, and titles
- **Collaborative Filtering**: Matrix factorization using SVD for user-item interactions
- **Hybrid Approach**: Combines both methods and is the main recommended mode for the assignment
- **Interactive Dashboard**: Web interface built with Dash for easy exploration
- **Scalable Design**: Handles large datasets with memory-efficient sampling

## Dataset

This project uses the MovieLens Latest Dataset (ml-latest):
- **86,537 movies** with genre information
- **33,832,162 ratings** from 330,975 users
- **2,328,315 tags** providing rich semantic information
- **19 unique genres** from Action to Western

## Installation

1. Clone or download this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Download and extract the MovieLens dataset (the script will do this automatically if you run `data_analysis.py` first)

## Usage

### Command Line Interface

Run the recommendation system directly:

```bash
python3 recommendation_system.py
```

This will:
- Load and analyze the dataset
- Train both content-based and collaborative filtering models
- Show sample recommendations for test movies
- Save the trained model as `movie_recommender.pkl`

### Interactive Dashboard

Start the web interface:

```bash
python3 app.py
```

Then open http://127.0.0.1:8050 in your browser to:
- Search for any movie
- Choose recommendation method (content-based, collaborative, or hybrid)
- Get 5 personalized recommendations
- View dataset statistics

### Data Analysis

Analyze the dataset structure and create visualizations:

```bash
python3 data_analysis.py
```

This generates:
- Dataset statistics and insights
- Visualizations saved as `dataset_analysis.png`
- Understanding of data distributions

## Methods Implemented

### 1. Content-Based Filtering

- **Feature Engineering**: Combines movie genres, user tags, and cleaned titles
- **TF-IDF Vectorization**: Captures term importance with n-grams (1,2)
- **Cosine Similarity**: Measures content similarity between movies
- **Pros**:
  - Works well with rich metadata such as genres and tags
  - Easy to explain in a report
  - Efficient to run locally
  - Good for movies with limited user-rating history
- **Cons**:
  - Can become too dependent on surface similarity
  - May recommend movies that look similar but are not especially well liked
  - Does not fully capture user preference patterns on its own

### 2. Collaborative Filtering

- **Matrix Factorization**: Uses Truncated SVD for dimensionality reduction
- **User-Item Matrix**: Sparse matrix of user ratings
- **Similarity Calculation**: Based on users who liked similar movies
- **Pros**:
  - Uses real user behavior from ratings
  - Can discover useful recommendations beyond simple genre overlap
  - Is a classic recommendation-system method and fits the assignment well
- **Cons**:
  - Ratings data is very large and memory-intensive
  - Sparse user-item matrices are harder to handle on a normal laptop
  - Weaker for movies with few ratings

### 3. Hybrid Approach

- **Weighted Combination**: Balances content similarity with rating and collaborative signals
- **Configurable Weights**: Adjusts content vs. collaborative influence in the dashboard
- **Rating-Aware Ranking**: Uses ratings to improve ranking quality among similar movies
- **Default**: 60% content, 40% collaborative weighting in the app
- **Pros**:
  - Combines the interpretability of content-based filtering with the strength of user feedback
  - Produces more practical recommendations than using metadata alone
  - Matches the assignment well because it shows both approaches and an improved combined technique
- **Cons**:
  - More complex than using a single model
  - Requires tuning of weights and fallback behavior
  - Can still be influenced by popularity bias if ratings are over-weighted
