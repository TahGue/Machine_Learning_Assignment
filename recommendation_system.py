#!/usr/bin/env python3


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import os
from collections import defaultdict
import re

class MovieRecommendationSystem:
    def __init__(self, data_path='ml-latest/'):
       
        self.data_path = data_path
        self.movies = None
        self.ratings = None
        self.tags = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.movie_features = None
        self.collaborative_matrix = None
        self.svd_model = None
        self.movie_rating_stats = None
        
    def load_data(self, sample_ratings=None):
        
        print("Loading MovieLens dataset...")
        
       
        self.movies = pd.read_csv(os.path.join(self.data_path, 'movies.csv'))
        print(f"Loaded {len(self.movies)} movies")
        
       
        ratings_path = os.path.join(self.data_path, 'ratings.csv')
        if sample_ratings:
            self.ratings = pd.read_csv(ratings_path, nrows=sample_ratings)
            print(f"Loaded {len(self.ratings)} ratings (sampled)")
        else:
           
            full_ratings = pd.read_csv(ratings_path, nrows=2000000)
           
            user_sample = full_ratings['userId'].drop_duplicates().sample(
                n=min(50000, full_ratings['userId'].nunique()), 
                
            )
            self.ratings = full_ratings[full_ratings['userId'].isin(user_sample)]
            print(f"Loaded {len(self.ratings)} ratings (strategic sample)")
        
       
        self.tags = pd.read_csv(os.path.join(self.data_path, 'tags.csv'))
        print(f"Loaded {len(self.tags)} tags")
        
       
        self._preprocess_data()
        
    def _preprocess_data(self):
       
        
        self.movies['title_clean'] = self.movies['title'].apply(self._clean_title)
        
       
        self.movies['genre_list'] = self.movies['genres'].apply(
            lambda x: [] if x == '(no genres listed)' else x.split('|')
        )
        
        
        self.tags['tag_clean'] = self.tags['tag'].apply(self._clean_tag)
        
        
        self._create_movie_features()
        self._create_rating_features()
        
    def _clean_title(self, title):
       
        
        title = re.sub(r'\(\d{4}\)', '', title)  
        title = re.sub(r'[^\w\s]', ' ', title)  
        return title.lower().strip()
    
    def _clean_tag(self, tag):
       
        if pd.isna(tag):
            return ""
        tag = str(tag).lower()
       
        tag = re.sub(r'[^\w\s]', ' ', tag)
        
        tag = re.sub(r'\s+', ' ', tag).strip()
        return tag
    
    def _create_movie_features(self):
       
        print("Creating movie features...")
        
        
        movie_tags = self.tags.groupby('movieId')['tag_clean'].apply(
            lambda x: ' '.join(x.dropna().unique())
        ).reset_index()
        movie_tags.columns = ['movieId', 'combined_tags']
        
       
        self.movie_features = self.movies.merge(movie_tags, on='movieId', how='left')
        self.movie_features['combined_tags'] = self.movie_features['combined_tags'].fillna('')
        
     
        self.movie_features['text_features'] = (
            self.movie_features['genre_list'].apply(lambda x: ' '.join(x)) + ' ' +
            self.movie_features['combined_tags'] + ' ' +
            self.movie_features['title_clean']
        )
        
        print("Movie features created successfully")

    def _create_rating_features(self):
       
        if self.ratings is None or self.ratings.empty:
            self.movie_rating_stats = pd.DataFrame(columns=['movieId', 'avg_rating', 'rating_count', 'weighted_rating'])
            return

        rating_stats = self.ratings.groupby('movieId')['rating'].agg(['mean', 'count']).reset_index()
        rating_stats.columns = ['movieId', 'avg_rating', 'rating_count']

        min_votes = max(5, int(rating_stats['rating_count'].quantile(0.60)))
        global_mean = rating_stats['avg_rating'].mean()

        rating_stats['weighted_rating'] = (
            (rating_stats['rating_count'] / (rating_stats['rating_count'] + min_votes)) * rating_stats['avg_rating']
            + (min_votes / (rating_stats['rating_count'] + min_votes)) * global_mean
        )

        self.movie_rating_stats = rating_stats
    
    def build_content_based_model(self):
        
        print("Building content-based model...")
        
       
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  
            min_df=2,  
            max_df=0.8 
        )
        
      
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.movie_features['text_features']
        )
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
    def build_collaborative_model(self, n_components=50):
       
        print("Building collaborative filtering model...")
        
        
        user_item_matrix = self.ratings.pivot(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        print(f"User-item matrix shape: {user_item_matrix.shape}")
        
        
        self.svd_model = TruncatedSVD(n_components=n_components)
        self.collaborative_matrix = self.svd_model.fit_transform(user_item_matrix)
        
       
        self.movie_collaborative_matrix = self.svd_model.components_.T
        
        print(f"Collaborative matrix shape: {self.collaborative_matrix.shape}")
        
    def get_content_recommendations(self, movie_title, n_recommendations=5):
      
        
        movie_idx = self.movies[
            self.movies['title'].str.contains(movie_title, case=False, na=False)
        ].index
        
        if len(movie_idx) == 0:
            return []
        
        movie_idx = movie_idx[0]
        
     
        movie_vector = self.tfidf_matrix[movie_idx:movie_idx+1]
        
        
        similarities = cosine_similarity(movie_vector, self.tfidf_matrix).flatten()

        candidates = self.movies[['movieId', 'title', 'genres']].copy()
        candidates['similarity_score'] = similarities

        if self.movie_rating_stats is not None and not self.movie_rating_stats.empty:
            candidates = candidates.merge(self.movie_rating_stats, on='movieId', how='left')
            candidates['avg_rating'] = candidates['avg_rating'].fillna(0)
            candidates['rating_count'] = candidates['rating_count'].fillna(0)
            candidates['weighted_rating'] = candidates['weighted_rating'].fillna(0)

            max_weighted_rating = candidates['weighted_rating'].max()
            if max_weighted_rating > 0:
                candidates['rating_score'] = candidates['weighted_rating'] / max_weighted_rating
            else:
                candidates['rating_score'] = 0
        else:
            candidates['avg_rating'] = 0.0
            candidates['rating_count'] = 0
            candidates['weighted_rating'] = 0.0
            candidates['rating_score'] = 0.0

        candidates['final_score'] = (0.8 * candidates['similarity_score']) + (0.2 * candidates['rating_score'])
        candidates = candidates.drop(index=movie_idx)
        top_candidates = candidates.sort_values('final_score', ascending=False).head(n_recommendations)
        
        recommendations = []
        for _, movie in top_candidates.iterrows():
            recommendations.append({
                'title': movie['title'],
                'genres': movie['genres'],
                'similarity_score': movie['similarity_score'],
                'avg_rating': movie['avg_rating'],
                'rating_count': int(movie['rating_count']),
                'final_score': movie['final_score']
            })
        
        return recommendations
    
    def get_collaborative_recommendations(self, movie_title, n_recommendations=5):
        
        if self.ratings is None or self.ratings.empty:
            return []

        movie_row = self.movies[
            self.movies['title'].str.contains(movie_title, case=False, na=False)
        ]
        
        if len(movie_row) == 0:
            return []
        
        movie_id = movie_row.iloc[0]['movieId']
        

        high_rating_users = self.ratings[
            (self.ratings['movieId'] == movie_id) & 
            (self.ratings['rating'] >= 4.0)
        ]['userId'].unique()
        
        if len(high_rating_users) == 0:
            return []
        
       
        user_ratings = self.ratings[
            self.ratings['userId'].isin(high_rating_users)
        ]
        
      
        movie_avg_ratings = user_ratings.groupby('movieId')['rating'].agg([
            'mean', 'count'
        ]).reset_index()
        
       
        popular_movies = movie_avg_ratings[movie_avg_ratings['count'] >= 5]
        
      
        popular_movies = popular_movies[popular_movies['movieId'] != movie_id]
        
       
        popular_movies = popular_movies.sort_values(
            ['mean', 'count'], 
            ascending=[False, False]
        )
        
       
        top_movies = popular_movies.head(n_recommendations)
        
        recommendations = []
        for _, row in top_movies.iterrows():
            movie = self.movies[self.movies['movieId'] == row['movieId']].iloc[0]
            recommendations.append({
                'title': movie['title'],
                'genres': movie['genres'],
                'avg_rating': row['mean'],
                'rating_count': row['count']
            })
        
        return recommendations
    
    def get_hybrid_recommendations(self, movie_title, n_recommendations=5, content_weight=0.6):
        
        content_recs = self.get_content_recommendations(movie_title, n_recommendations * 2)
        collaborative_recs = self.get_collaborative_recommendations(movie_title, n_recommendations * 2)

        if not collaborative_recs:
            return content_recs[:n_recommendations]
        if not content_recs:
            return collaborative_recs[:n_recommendations]
        
       
        movie_scores = defaultdict(float)
        movie_info = {}
        
      
        for rec in content_recs:
            movie_scores[rec['title']] += content_weight * rec['similarity_score']
            movie_info[rec['title']] = rec
        
       
        for rec in collaborative_recs:
            normalized_score = rec['avg_rating'] / 5.0
            movie_scores[rec['title']] += (1 - content_weight) * normalized_score
            movie_info[rec['title']] = rec
        
       
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for title, score in sorted_movies[:n_recommendations]:
            rec = movie_info[title].copy()
            rec['combined_score'] = score
            recommendations.append(rec)
        
        return recommendations
    
    def save_model(self, filename='movie_recommender.pkl'):
      
        model_data = {
            'movies': self.movies,
            'ratings': self.ratings,
            'tags': self.tags,
            'movie_features': self.movie_features,
            'movie_rating_stats': self.movie_rating_stats,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix,
            'svd_model': self.svd_model,
            'movie_collaborative_matrix': self.movie_collaborative_matrix
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='movie_recommender.pkl'):
       
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        self.movies = model_data['movies']
        self.ratings = model_data.get('ratings')
        self.tags = model_data.get('tags')
        self.movie_features = model_data['movie_features']
        self.movie_rating_stats = model_data.get('movie_rating_stats')
        self.tfidf_vectorizer = model_data['tfidf_vectorizer']
        self.tfidf_matrix = model_data['tfidf_matrix']
        self.svd_model = model_data['svd_model']
        self.movie_collaborative_matrix = model_data['movie_collaborative_matrix']
        
        print(f"Model loaded from {filename}")
    
    def search_movies(self, query, limit=10):
        
        matches = self.movies[
            self.movies['title'].str.contains(query, case=False, na=False)
        ].head(limit)
        
        return matches[['movieId', 'title', 'genres']].to_dict('records')

def main():
    
    print("Movie Recommendation System")
    print("=" * 50)
    
   
    recommender = MovieRecommendationSystem()
    
    
    recommender.load_data(sample_ratings=1000000)
  
    recommender.build_content_based_model()
    recommender.build_collaborative_model()
    
   
    test_movies = ["Toy Story", "Matrix", "Star Wars", "Pulp Fiction"]
    
    for movie in test_movies:
        print(f"\n--- Recommendations for '{movie}' ---")
        
        
        content_recs = recommender.get_content_recommendations(movie, 3)
        print("Content-based:")
        for i, rec in enumerate(content_recs, 1):
            print(f"  {i}. {rec['title']} ({rec['similarity_score']:.3f})")
        
      
        collab_recs = recommender.get_collaborative_recommendations(movie, 3)
        print("Collaborative:")
        for i, rec in enumerate(collab_recs, 1):
            print(f"  {i}. {rec['title']} ({rec['avg_rating']:.2f}, {rec['rating_count']} ratings)")
        
       
        hybrid_recs = recommender.get_hybrid_recommendations(movie, 3)
        print("Hybrid:")
        for i, rec in enumerate(hybrid_recs, 1):
            print(f"  {i}. {rec['title']} ({rec['combined_score']:.3f})")
    
    
    recommender.save_model()

if __name__ == "__main__":
    main()
