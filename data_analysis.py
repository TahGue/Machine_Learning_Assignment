#!/usr/bin/env python3


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

def load_datasets():
   
    print("Loading datasets...")
    
    
    movies = pd.read_csv('ml-latest/movies.csv')
    print(f"Movies: {len(movies)} movies")
    
   
    print("Loading ratings (sampling 1M rows for analysis)...")
    ratings_sample = pd.read_csv('ml-latest/ratings.csv', nrows=1000000)
    print(f"Ratings sample: {len(ratings_sample)} ratings")
    
  
    print("Loading tags (sampling 100K rows for analysis)...")
    tags_sample = pd.read_csv('ml-latest/tags.csv', nrows=100000)
    print(f"Tags sample: {len(tags_sample)} tags")
    
    return movies, ratings_sample, tags_sample

def analyze_movies(movies):
   
    print("\n=== MOVIES ANALYSIS ===")
    
   
    all_genres = []
    for genres in movies['genres']:
        if genres != '(no genres listed)':
            all_genres.extend(genres.split('|'))
    
    genre_counts = Counter(all_genres)
    print(f"Total unique genres: {len(genre_counts)}")
    print("Top 10 genres:")
    for genre, count in genre_counts.most_common(10):
        print(f"  {genre}: {count}")
    
   
    years = []
    for title in movies['title']:
        year_match = title.split('(')[-1].split(')')[0] if '(' in title else ''
        if year_match.isdigit() and len(year_match) == 4:
            years.append(int(year_match))
    
    if years:
        print(f"\nYear range: {min(years)} - {max(years)}")
        print(f"Median year: {np.median(years)}")
    
    return genre_counts, years

def analyze_ratings(ratings):
    """Analyze the ratings dataset"""
    print("\n=== RATINGS ANALYSIS ===")
    
    
    rating_dist = ratings['rating'].value_counts().sort_index()
    print("Rating distribution:")
    for rating, count in rating_dist.items():
        percentage = (count / len(ratings)) * 100
        print(f"  {rating}: {count} ({percentage:.1f}%)")
    
  
    print(f"\nRating statistics:")
    print(f"  Mean: {ratings['rating'].mean():.3f}")
    print(f"  Median: {ratings['rating'].median():.3f}")
    print(f"  Std: {ratings['rating'].std():.3f}")
  
    unique_users = ratings['userId'].nunique()
    unique_movies = ratings['movieId'].nunique()
    print(f"\nUnique users: {unique_users}")
    print(f"Unique movies: {unique_movies}")
    print(f"Average ratings per user: {len(ratings) / unique_users:.1f}")
    print(f"Average ratings per movie: {len(ratings) / unique_movies:.1f}")
    
    return rating_dist

def analyze_tags(tags):
    """Analyze the tags dataset"""
    print("\n=== TAGS ANALYSIS ===")
    
 
    tag_counts = tags['tag'].value_counts()
    print(f"Total unique tags: {len(tag_counts)}")
    print("Top 10 tags:")
    for tag, count in tag_counts.head(10).items():
        print(f"  '{tag}': {count}")
    
   
    tag_lengths = tags['tag'].str.len()
    print(f"\nAverage tag length: {tag_lengths.mean():.1f} characters")
    
    return tag_counts

def create_visualizations(movies, ratings, genre_counts):
    """Create visualizations for the dataset"""
    print("\nCreating visualizations...")
    
    
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MovieLens Dataset Analysis', fontsize=16, fontweight='bold')
    

    top_genres = dict(genre_counts.most_common(15))
    axes[0, 0].barh(list(top_genres.keys()), list(top_genres.values()))
    axes[0, 0].set_title('Top 15 Movie Genres')
    axes[0, 0].set_xlabel('Number of Movies')
    
  
    rating_dist = ratings['rating'].value_counts().sort_index()
    axes[0, 1].bar(rating_dist.index, rating_dist.values, color='skyblue')
    axes[0, 1].set_title('Rating Distribution')
    axes[0, 1].set_xlabel('Rating')
    axes[0, 1].set_ylabel('Count')
    
   
    years = []
    for title in movies['title']:
        year_match = title.split('(')[-1].split(')')[0] if '(' in title else ''
        if year_match.isdigit() and len(year_match) == 4:
            year = int(year_match)
            if year >= 1970: 
                years.append(year)
    
    if years:
        year_counts = Counter(years)
        years_sorted = sorted(year_counts.items())
        axes[1, 0].plot([y[0] for y in years_sorted], [y[1] for y in years_sorted])
        axes[1, 0].set_title('Movies Released Per Year (1970+)')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Number of Movies')
    
   
    user_counts = ratings['userId'].value_counts()
    axes[1, 1].hist(user_counts.values, bins=50, alpha=0.7, color='lightcoral')
    axes[1, 1].set_title('User Rating Activity Distribution')
    axes[1, 1].set_xlabel('Number of Ratings per User')
    axes[1, 1].set_ylabel('Number of Users')
    axes[1, 1].set_xlim(0, 200)
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'dataset_analysis.png'")

def main():
    """Main analysis function"""
    print("MovieLens Dataset Analysis")
    print("=" * 50)
    
   
    movies, ratings, tags = load_datasets()
    
   
    genre_counts, years = analyze_movies(movies)
    rating_dist = analyze_ratings(ratings)
    tag_counts = analyze_tags(tags)
    
  
    create_visualizations(movies, ratings, genre_counts)
    
    print("\n" + "=" * 50)
    print("Analysis complete!")
    print("Key insights:")
    print(f"- Dataset contains {len(movies)} movies with {len(genre_counts)} unique genres")
    print(f"- Rating distribution shows user preference (check visualization)")
    print(f"- Tags provide rich semantic information for content-based filtering")
    print(f"- Consider using TF-IDF on combined genres + tags for better recommendations")

if __name__ == "__main__":
    main()
