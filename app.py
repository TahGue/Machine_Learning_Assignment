#!/usr/bin/env python3

import dash
from dash import dcc, html, Input, Output, callback, State, ALL
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re
import json
from recommendation_system import MovieRecommendationSystem
import pickle
import os

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap"
    ],
    suppress_callback_exceptions=True
)

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


# ─── Global custom CSS ───────────────────────────────────────────────────────
CUSTOM_CSS = """
  :root {
    --bg-deep:    #0a0b0f;
    --bg-panel:   #11131a;
    --bg-card:    #181b25;
    --bg-hover:   #1f2335;
    --accent:     #e8c97d;
    --accent-dim: #a8904d;
    --text-primary: #edeaf2;
    --text-muted:   #7a7f96;
    --border:     rgba(232,201,125,0.15);
    --border-hover: rgba(232,201,125,0.45);
    --red:        #e07060;
    --green:      #6ec98f;
    --blue:       #6a9fd8;
  }

  * { box-sizing: border-box; }

  body {
    background: var(--bg-deep);
    color: var(--text-primary);
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    min-height: 100vh;
    background-image:
      radial-gradient(ellipse 80% 50% at 50% -10%, rgba(232,201,125,0.07) 0%, transparent 70%),
      radial-gradient(ellipse 40% 30% at 90% 80%, rgba(106,159,216,0.05) 0%, transparent 60%);
  }

  /* ── Header ── */
  .site-header {
    border-bottom: 1px solid var(--border);
    padding: 36px 0 28px;
    text-align: center;
    position: relative;
  }
  .site-header::after {
    content: '';
    position: absolute;
    bottom: -1px; left: 50%; transform: translateX(-50%);
    width: 120px; height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
  }
  .main-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2rem, 5vw, 3.4rem);
    letter-spacing: -0.5px;
    color: var(--text-primary);
    margin: 0 0 8px;
    line-height: 1.1;
  }
  .main-title span { color: var(--accent); font-style: italic; }
  .main-subtitle {
    font-size: 0.95rem;
    color: var(--text-muted);
    font-weight: 300;
    letter-spacing: 0.5px;
  }

  /* ── Panels ── */
  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 28px 28px 24px;
    transition: border-color 0.3s;
  }
  .panel:hover { border-color: var(--border-hover); }
  .panel-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 16px;
  }

  /* ── Search Input ── */
  #movie-search {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 13px 18px !important;
    transition: border-color 0.25s, box-shadow 0.25s;
    outline: none !important;
  }
  #movie-search:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(232,201,125,0.12) !important;
  }
  #movie-search::placeholder { color: var(--text-muted) !important; }

  /* ── Primary Button ── */
  #recommend-btn {
    background: var(--accent) !important;
    border: none !important;
    border-radius: 10px !important;
    color: #0a0b0f !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.4px !important;
    padding: 13px 22px !important;
    transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
    width: 100% !important;
  }
  #recommend-btn:hover {
    background: #f0d78e !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(232,201,125,0.28) !important;
  }
  #recommend-btn:active { transform: translateY(0) !important; }

  /* ── Suggestions ── */
  .suggestion-item {
    padding: 9px 14px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.88rem;
    color: var(--text-muted);
    transition: background 0.2s, color 0.2s;
    border: 1px solid transparent;
  }
  .suggestion-item:hover {
    background: var(--bg-hover);
    color: var(--text-primary);
    border-color: var(--border);
  }

  /* ── Method pills ── */
  .method-pill {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 9px 20px;
    cursor: pointer;
    font-size: 0.85rem;
    color: var(--text-muted);
    font-weight: 500;
    transition: all 0.2s;
    user-select: none;
  }
  .method-pill:hover { border-color: var(--accent); color: var(--text-primary); }
  .method-pill.active {
    background: rgba(232,201,125,0.12);
    border-color: var(--accent);
    color: var(--accent);
  }

  /* ── Slider overrides ── */
  .rc-slider-track { background: var(--accent) !important; }
  .rc-slider-handle {
    border-color: var(--accent) !important;
    background: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(232,201,125,0.2) !important;
  }
  .rc-slider-rail { background: var(--bg-hover) !important; }

  /* ── Rec Cards ── */
  .rec-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 22px 20px;
    height: 100%;
    transition: all 0.28s cubic-bezier(.22,.68,0,1.2);
    position: relative;
    overflow: hidden;
  }
  .rec-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--accent), transparent);
    opacity: 0;
    transition: opacity 0.3s;
  }
  .rec-card:hover {
    border-color: var(--border-hover);
    transform: translateY(-5px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.5);
  }
  .rec-card:hover::before { opacity: 1; }

  .rec-rank {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    color: rgba(232,201,125,0.2);
    line-height: 1;
    margin-bottom: 4px;
    font-style: italic;
  }
  .rec-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: var(--text-primary);
    margin-bottom: 6px;
    line-height: 1.3;
  }
  .rec-genres {
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-bottom: 14px;
  }
  .score-bar-wrap {
    background: var(--bg-hover);
    border-radius: 4px;
    height: 4px;
    overflow: hidden;
    margin-bottom: 10px;
  }
  .score-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-dim), var(--accent));
    transition: width 0.8s cubic-bezier(.22,.68,0,1.2);
  }
  .score-label {
    font-size: 0.78rem;
    color: var(--accent);
    font-weight: 600;
  }
  .method-badge {
    display: inline-block;
    background: rgba(232,201,125,0.1);
    border: 1px solid rgba(232,201,125,0.25);
    border-radius: 50px;
    padding: 3px 12px;
    font-size: 0.72rem;
    color: var(--accent);
    font-weight: 500;
    margin-top: 12px;
  }

  /* ── Genre tags ── */
  .genre-tag {
    display: inline-block;
    background: var(--bg-hover);
    border: 1px solid var(--border);
    border-radius: 50px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: var(--text-muted);
    margin: 2px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .genre-tag:hover {
    background: rgba(232,201,125,0.1);
    border-color: var(--accent);
    color: var(--accent);
  }

  /* ── Stats ── */
  .stat-block {
    text-align: center;
    padding: 14px 0;
  }
  .stat-number {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--accent);
    display: block;
    line-height: 1;
    margin-bottom: 4px;
  }
  .stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
  }
  .stat-divider {
    border: none;
    border-left: 1px solid var(--border);
    height: 48px;
    margin: auto;
  }

  /* ── Alert ── */
  .custom-alert {
    background: rgba(224,112,96,0.1);
    border: 1px solid rgba(224,112,96,0.3);
    border-radius: 10px;
    padding: 18px 22px;
    color: var(--red);
    font-size: 0.9rem;
  }

  /* ── Footer ── */
  .site-footer {
    border-top: 1px solid var(--border);
    margin-top: 60px;
    padding: 24px 0;
    text-align: center;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.5px;
  }

  /* ── Spinner override ── */
  ._dash-loading { background: none !important; }
  ._dash-loading-callback { opacity: 0.5 !important; transition: opacity 0.4s; }

  /* ── Plotly chart ── */
  .js-plotly-plot .plotly { background: transparent !important; }
"""

# Inject CSS via index_string — the correct approach for all Dash versions
app.index_string = '''
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>Movie Recommender</title>
    {%favicon%}
    {%css%}
    <style>''' + CUSTOM_CSS + '''</style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
'''


# ─── Layout ──────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # ── Header ──────────────────────────────────────────────────────────────
    html.Header(className="site-header", children=[
        dbc.Container([
            html.H1(["🎬 Movie ", html.Span("Recommender")], className="main-title"),
            html.P(
                "Discover your next favourite film — powered by machine learning.",
                className="main-subtitle"
            )
        ], fluid=True)
    ]),

    dbc.Container(fluid=True, style={"maxWidth": "1280px", "padding": "0 24px"}, children=[

        html.Div(style={"height": "40px"}),

        # ── Search Row ───────────────────────────────────────────────────────
        dcc.Store(id="suggestions-store", data=[]),

        dbc.Row([
            dbc.Col(md=12, children=[
                html.Div(className="panel", children=[
                    html.P("Find a movie", className="panel-label"),
                    dbc.Row([
                        dbc.Col(width=10, children=[
                            dbc.Input(
                                id="movie-search",
                                type="text",
                                placeholder='e.g. "Toy Story", "The Matrix"…',
                                value="Toy Story",
                            )
                        ]),
                        dbc.Col(width=2, children=[
                            dbc.Button("Recommend →", id="recommend-btn", n_clicks=0)
                        ])
                    ], className="g-2"),
                    html.Div(id="movie-suggestions", style={"marginTop": "12px"})
                ])
            ]),
        ], className="g-4 mb-4"),

        # ── Recommendations output ───────────────────────────────────────────
        dcc.Loading(
            id="loading",
            type="dot",
            color="var(--accent)",
            children=[html.Div(id="recommendations-output")]
        ),

        html.Div(style={"height": "48px"}),

        # ── Stats + Genre chart Row ──────────────────────────────────────────
        dbc.Row([
            dbc.Col(md=5, children=[
                html.Div(className="panel", children=[
                    html.P("Dataset Overview", className="panel-label"),
                    html.Div(id="dataset-stats")
                ])
            ]),
            dbc.Col(md=7, children=[
                html.Div(className="panel", children=[
                    html.P("Genre Distribution", className="panel-label"),
                    dcc.Graph(
                        id="genre-chart",
                        config={"displayModeBar": False},
                        style={"height": "260px"}
                    )
                ])
            ])
        ], className="g-4"),

        # ── Footer ──────────────────────────────────────────────────────────
        html.Footer(className="site-footer", children=[
            "Built with the MovieLens dataset  ·  Machine Learning Assignment 1"
        ])
    ])
])


# ─── Callbacks ───────────────────────────────────────────────────────────────

# Suggestions
@callback(
    Output("movie-suggestions", "children"),
    Output("suggestions-store", "data"),
    Input("movie-search", "value")
)
def update_suggestions(search_term):
    if not search_term or len(search_term) < 2:
        return "", []
    try:
        suggestions = recommender.search_movies(search_term, limit=5)
        if suggestions:
            titles = [s['title'] for s in suggestions]
            # Strip year for clean display in input e.g. "Toy Story" not "Toy Story (1995)"
            clean_titles = [re.sub(r'\s*\(\d{4}\)\s*$', '', t).strip() for t in titles]
            items = html.Div([
                html.Div(
                    f"▸  {s['title']}  ·  {s['genres']}",
                    className="suggestion-item",
                    id={"type": "suggestion", "index": i},
                    n_clicks=0
                )
                for i, s in enumerate(suggestions)
            ])
            return items, clean_titles
    except Exception:
        return html.Small("Could not load suggestions.", style={"color": "var(--text-muted)"}), []
    return "", []


# Click a suggestion → fill input and clear dropdown
@callback(
    Output("movie-search", "value"),
    Output("movie-suggestions", "children", allow_duplicate=True),
    Input({"type": "suggestion", "index": ALL}, "n_clicks"),
    State("suggestions-store", "data"),
    prevent_initial_call=True
)
def select_suggestion(n_clicks_list, titles):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list):
        raise dash.exceptions.PreventUpdate
    # Find which suggestion was clicked
    triggered = ctx.triggered[0]["prop_id"]
    idx = json.loads(triggered.split(".")[0])["index"]
    if titles and idx < len(titles):
        return titles[idx], ""
    raise dash.exceptions.PreventUpdate


# Main recommendations
@callback(
    Output("recommendations-output", "children"),
    Input("recommend-btn", "n_clicks"),
    State("movie-search", "value"),
    prevent_initial_call=True
)
def get_recommendations(n_clicks, movie_title):
    method = "hybrid"
    content_weight = 0.6
    if not movie_title:
        return ""
    try:
        recs = recommender.get_hybrid_recommendations(movie_title, 5, content_weight)
        if not recs:
            recs = recommender.get_content_recommendations(movie_title, 5)
            method = "content_fallback"

        if not recs:
            return html.Div(className="custom-alert", children=[
                html.Strong("No results found."),
                html.Span("  Try a different title.")
            ])

        method_label = {
            "content": "Content-Based",
            "collaborative": "Collaborative",
            "hybrid": "Hybrid",
            "content_fallback": "Content-Based (fallback)"
        }[method]

        cards = []
        for i, rec in enumerate(recs, 1):
            score_key = next(
                (k for k in ("similarity_score", "combined_score", "avg_rating") if k in rec), None
            )
            score = rec[score_key] if score_key else 0
            score_pct = min(100, int(score * 100)) if score <= 1 else min(100, int(score / 5 * 100))
            score_disp = (
                f"Similarity {score:.3f}" if method in ("content", "content_fallback") else
                f"Rating {score:.2f} / 5  ({rec.get('rating_count', 0)} ratings)" if method == "collaborative" else
                f"Score {score:.3f}"
            )

            cards.append(
                dbc.Col(lg=4, md=6, xs=12, style={"marginBottom": "20px"}, children=[
                    html.Div(className="rec-card", children=[
                        html.Div(str(i).zfill(2), className="rec-rank"),
                        html.Div(rec["title"], className="rec-title"),
                        html.Div(rec["genres"], className="rec-genres"),
                        html.Div(className="score-bar-wrap", children=[
                            html.Div(className="score-bar",
                                     style={"width": f"{score_pct}%"})
                        ]),
                        html.Div(score_disp, className="score-label"),
                        html.Div(f"⊙ {method_label}", className="method-badge")
                    ])
                ])
            )

        return html.Div([
            html.P(
                f"Showing 5 recommendations for \"{movie_title}\" via {method_label}",
                style={"fontSize": "0.85rem", "color": "var(--text-muted)",
                       "marginBottom": "20px", "fontWeight": "400"}
            ),
            dbc.Row(cards, className="g-0")
        ])

    except Exception as e:
        return html.Div(className="custom-alert", children=[
            html.Strong("Error: "), str(e)
        ])


# Stats + genre chart
@callback(
    Output("dataset-stats", "children"),
    Output("genre-chart", "figure"),
    Input("recommend-btn", "n_clicks")
)
def update_stats(n_clicks):
    if recommender.movies is None:
        return "Loading…", go.Figure()

    total_movies  = len(recommender.movies)
    total_ratings = len(recommender.ratings) if recommender.ratings is not None else 0
    total_users   = recommender.ratings["userId"].nunique() if recommender.ratings is not None else 0
    total_tags    = len(recommender.tags) if recommender.tags is not None else 0

    stats_html = dbc.Row([
        dbc.Col(children=[
            html.Div(className="stat-block", children=[
                html.Span(f"{total_movies:,}", className="stat-number"),
                html.Span("Movies", className="stat-label")
            ])
        ], xs=6, md=3),
        dbc.Col(children=[
            html.Div(className="stat-block", children=[
                html.Span(f"{total_ratings:,}", className="stat-number"),
                html.Span("Ratings", className="stat-label")
            ])
        ], xs=6, md=3),
        dbc.Col(children=[
            html.Div(className="stat-block", children=[
                html.Span(f"{total_users:,}", className="stat-number"),
                html.Span("Users", className="stat-label")
            ])
        ], xs=6, md=3),
        dbc.Col(children=[
            html.Div(className="stat-block", children=[
                html.Span(f"{total_tags:,}", className="stat-number"),
                html.Span("Tags", className="stat-label")
            ])
        ], xs=6, md=3),
    ], className="g-0")

    # Genre chart
    all_genres = []
    for genres in recommender.movies["genres"]:
        if genres != "(no genres listed)":
            all_genres.extend(genres.split("|"))

    genre_counts = pd.Series(all_genres).value_counts().head(12)

    fig = go.Figure(go.Bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation="h",
        marker=dict(
            color=genre_counts.values,
            colorscale=[[0, "#2a2210"], [1, "#e8c97d"]],
            line=dict(width=0)
        ),
        hovertemplate="%{y}: %{x:,}<extra></extra>"
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8),
        font=dict(family="DM Sans", color="#7a7f96", size=11),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   showline=False, tickfont=dict(color="#7a7f96")),
        yaxis=dict(showgrid=False, showline=False,
                   tickfont=dict(color="#edeaf2", size=12), autorange="reversed"),
        hoverlabel=dict(bgcolor="#181b25", bordercolor="#e8c97d",
                        font=dict(color="#edeaf2"))
    )

    return stats_html, fig


if __name__ == "__main__":
    print("Starting Movie Recommendation Dashboard…")
    print("Open http://127.0.0.1:8050 in your browser")
    app.run_server(debug=True, host="127.0.0.1", port=8050)