#!/usr/bin/env python3


import dash
from dash import dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from recommendation_system import MovieRecommendationSystem
import pickle
import os


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


recommender = MovieRecommendationSystem()


if os.path.exists('movie_recommender.pkl'):
    print("Loading pre-trained model...")
    recommender.load_model('movie_recommender.pkl')
else:
    print("Training new model...")
    recommender.load_data(sample_ratings=1000000)
    recommender.build_content_based_model()
    recommender.build_collaborative_model()
    recommender.save_model()


app.layout = dbc.Container([
   
    dbc.Row([
        dbc.Col([
            html.H1("🎬 Movie Recommendation System", 
                   className="text-center mb-4",
                   style={'color': '#2c3e50'}),
            html.P("Discover your next favorite movie using AI-powered recommendations",
                   className="text-center text-muted mb-4")
        ])
    ]),
    
   
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🔍 Search for a Movie", className="card-title"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Input(
                                id='movie-search',
                                type='text',
                                placeholder='Enter movie title (e.g., "Toy Story", "Matrix")',
                                value='Toy Story',
                                className="mb-3"
                            )
                        ], width=8),
                        dbc.Col([
                            dbc.Button(
                                "Get Recommendations",
                                id="recommend-btn",
                                color="primary",
                                className="w-100"
                            )
                        ], width=4)
                    ]),
                    
                    
                    html.Div(id='movie-suggestions')
                ])
            ], className="mb-4")
        ])
    ]),
    
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("⚙️ Recommendation Method", className="card-title"),
                    dbc.RadioItems(
                        id="method-selector",
                        options=[
                            {"label": "Content-Based (Genres + Tags)", "value": "content"},
                            {"label": "Collaborative Filtering (User Preferences)", "value": "collaborative"},
                            {"label": "Hybrid (Best of Both)", "value": "hybrid"}
                        ],
                        value="hybrid",
                        inline=True,
                        className="mb-3"
                    ),
                    
                    
                    html.Div([
                        html.Label("Content Weight:", className="form-label"),
                        dcc.Slider(
                            id='content-weight',
                            min=0,
                            max=1,
                            step=0.1,
                            value=0.6,
                            marks={i: f"{i}" for i in [0, 0.5, 1]},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], id='weight-slider', style={'display': 'none'})
                ])
            ], className="mb-4")
        ])
    ]),
    
   
    dbc.Row([
        dbc.Col([
            dcc.Loading(
                id="loading-spinner",
                children=[html.Div(id="loading-output")],
                type="default"
            )
        ])
    ]),
    
    
    dbc.Row([
        dbc.Col([
            html.Div(id="recommendations-output")
        ])
    ]),
    
   
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Dataset Statistics", className="card-title"),
                    html.Div(id="dataset-stats")
                ])
            ])
        ])
    ], className="mt-4"),
    
    
    dbc.Row([
        dbc.Col([
            html.P("Built with MovieLens dataset | Machine Learning Assignment 1",
                   className="text-center text-muted mt-4")
        ])
    ])
], fluid=True)


@callback(
    Output('movie-suggestions', 'children'),
    Input('movie-search', 'value')
)
def update_movie_suggestions(search_term):
    if not search_term or len(search_term) < 2:
        return ""
    
    try:
        suggestions = recommender.search_movies(search_term, limit=5)
        if suggestions:
            suggestion_list = [
                html.Div([
                    html.Small(f"• {sug['title']} ({sug['genres']})", 
                             className="text-muted",
                             style={'cursor': 'pointer'},
                             n_clicks=0,
                             id={'type': 'suggestion', 'index': i})
                ]) for i, sug in enumerate(suggestions)
            ]
            return html.Div(suggestion_list, className="mt-2")
    except Exception as e:
        return html.Small("Error loading suggestions", className="text-danger")
    
    return ""


@callback(
    Output('weight-slider', 'style'),
    Input('method-selector', 'value')
)
def toggle_weight_slider(method):
    if method == 'hybrid':
        return {'display': 'block'}
    return {'display': 'none'}


@callback(
    Output("recommendations-output", "children"),
    Output("loading-output", "children"),
    Input("recommend-btn", "n_clicks"),
    State("movie-search", "value"),
    State("method-selector", "value"),
    State("content-weight", "value"),
    prevent_initial_call=True
)
def get_recommendations(n_clicks, movie_title, method, content_weight):
    if not n_clicks or not movie_title:
        return "", ""
    
    try:
        
        if method == "content":
            recommendations = recommender.get_content_recommendations(movie_title, 5)
        elif method == "collaborative":
            recommendations = recommender.get_collaborative_recommendations(movie_title, 5)
        else:  # hybrid
            recommendations = recommender.get_hybrid_recommendations(
                movie_title, 5, content_weight
            )
        
        if not recommendations:
            return dbc.Alert([
                html.H4("No Recommendations Found", className="alert-heading"),
                html.P("Try searching for a different movie title.")
            ], color="warning"), ""
        
        
        recommendation_cards = []
        for i, rec in enumerate(recommendations, 1):
            score_key = 'similarity_score' if 'similarity_score' in rec else \
                       'avg_rating' if 'avg_rating' in rec else 'combined_score'
            score_value = rec[score_key]
            
            
            if method == "content":
                score_text = f"Similarity: {score_value:.3f}"
            elif method == "collaborative":
                score_text = f"Rating: {score_value:.2f} ({rec.get('rating_count', 0)} ratings)"
            else:
                score_text = f"Score: {score_value:.3f}"
            
            card = dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5(f"{i}. {rec['title']}", className="card-title"),
                        html.P(rec['genres'], className="text-muted mb-2"),
                        dbc.Badge(score_text, color="primary"),
                        html.P([
                            html.Small("⭐ Recommended based on "),
                            html.Strong(method.replace("-", " ").title())
                        ], className="mt-2 mb-0")
                    ])
                ], className="h-100")
            ], width=12, md=6, lg=4, className="mb-3")
            
            recommendation_cards.append(card)
        
        recommendations_html = [
            html.H4("🎯 Recommendations", className="mb-3"),
            dbc.Row(recommendation_cards)
        ]
        
        return recommendations_html, ""
        
    except Exception as e:
        return dbc.Alert([
            html.H4("Error", className="alert-heading"),
            html.P(f"An error occurred: {str(e)}")
        ], color="danger"), ""


@callback(
    Output("dataset-stats", "children"),
    Input("recommend-btn", "n_clicks")
)
def update_dataset_stats(n_clicks):
    if recommender.movies is None:
        return "Loading..."
    
    
    total_movies = len(recommender.movies)
    total_ratings = len(recommender.ratings) if recommender.ratings is not None else 0
    total_users = recommender.ratings['userId'].nunique() if recommender.ratings is not None else 0
    total_tags = len(recommender.tags) if recommender.tags is not None else 0
    
    
    all_genres = []
    for genres in recommender.movies['genres']:
        if genres != '(no genres listed)':
            all_genres.extend(genres.split('|'))
    
    genre_counts = pd.Series(all_genres).value_counts().head(5)
    
    stats_html = [
        dbc.Row([
            dbc.Col([
                html.H6(f"{total_movies:,}", className="display-6"),
                html.P("Movies", className="text-muted")
            ], width=3),
            dbc.Col([
                html.H6(f"{total_ratings:,}", className="display-6"),
                html.P("Ratings", className="text-muted")
            ], width=3),
            dbc.Col([
                html.H6(f"{total_users:,}", className="display-6"),
                html.P("Users", className="text-muted")
            ], width=3),
            dbc.Col([
                html.H6(f"{total_tags:,}", className="display-6"),
                html.P("Tags", className="text-muted")
            ], width=3)
        ], className="text-center mb-3"),
        
        html.H6("Top Genres:", className="mt-3"),
        html.Div([
            dbc.Badge(f"{genre}: {count}", color="secondary", className="me-2 mb-1")
            for genre, count in genre_counts.items()
        ])
    ]
    
    return stats_html


if __name__ == "__main__":
    print("Starting Movie Recommendation Dashboard...")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run_server(debug=True, host="127.0.0.1", port=8050)
