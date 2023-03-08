from flask import Flask, render_template
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Scale the input features
scaler = StandardScaler()

app = Flask(__name__)

@app.before_first_request
def load_data():
    # Load the data
    matches = pd.concat([
        pd.read_csv('soccer-spi/spi_matches.csv'),
        pd.read_csv('soccer-spi/spi_matches_latest.csv'),
        pd.read_csv('soccer-spi/spi_matches_intl.csv')
    ])

    rankings = pd.concat([
        pd.read_csv('soccer-spi/spi_global_rankings.csv'),
        pd.read_csv('soccer-spi/spi_global_rankings_intl.csv')
    ])

    # Preprocess and engineer features
    matches.drop(['league', 'season', 'score1', 'score2', 'importance1', 'importance2'], axis=1, inplace=True)
    matches['date'] = pd.to_datetime(matches['date'])
    matches['year'] = matches['date'].dt.year
    matches['month'] = matches['date'].dt.month
    matches['day'] = matches['date'].dt.day
    matches['weekday'] = matches['date'].dt.weekday
    matches['home_win'] = (matches['proj_score1'] > matches['proj_score2']).astype(int)


    # Merge the rankings with the matches
    home_rankings = rankings[['rank', 'name', 'spi']].rename(columns={'rank': 'home_rank', 'name': 'team1', 'spi': 'spi1'})
    away_rankings = rankings[['rank', 'name', 'spi']].rename(columns={'rank': 'away_rank', 'name': 'team2', 'spi': 'spi2'})
    matches = matches.merge(home_rankings, on=['team1'], how='left')
    matches = matches.merge(away_rankings, on=['team2'], how='left')
    matches.fillna({'home_rank': matches['away_rank'].max() + 1, 'away_rank': matches['home_rank'].max() + 1, 'spi1': 0, 'spi2': 0}, inplace=True)

    # Train the model
    # add to features: 'spi1', 'spi2', 'logprob1', 'logprob2', 'logprobtie',
    features = [ 'prob1', 'prob2', 'probtie',  'home_rank', 'away_rank', 'year', 'month', 'day', 'weekday']
    target = 'home_win'
    model = LogisticRegression(max_iter=1000000000000000)
    model.fit(matches[features], matches[target])

    # Store the model in the application context
    app.config['model'] = model
    app.config['features'] = features

@app.route('/')
def predict_scores():
    # Load the upcoming games
    url = 'https://projects.fivethirtyeight.com/soccer-api/club/spi_matches_latest.csv'
    upcoming = pd.read_csv(url)

    # Preprocess and engineer features
    upcoming['date'] = pd.to_datetime(upcoming['date'])
    upcoming['year'] = upcoming['date'].dt.year
    upcoming['month'] = upcoming['date'].dt.month
    upcoming['day'] = upcoming['date'].dt.day
    upcoming['weekday'] = upcoming['date'].dt.weekday

    # Merge the rankings with the upcoming games
    rankings = pd.concat([
        pd.read_csv('soccer-spi/spi_global_rankings.csv'),
        pd.read_csv('soccer-spi/spi_global_rankings_intl.csv')
    ])
    home_rankings = rankings[['rank', 'name', 'spi']].rename(columns={'rank': 'home_rank', 'name': 'team1', 'spi': 'spi1'})
    away_rankings = rankings[['rank', 'name', 'spi']].rename(columns={'rank': 'away_rank', 'name': 'team2', 'spi': 'spi2'})
    upcoming = upcoming.merge(home_rankings, on=['team1'], how='left')
    upcoming = upcoming.merge(away_rankings, on=['team2'], how='left')
    upcoming.fillna({'home_rank': upcoming['away_rank'].max() + 1, 'away_rank': upcoming['home_rank'].max() + 1, 'spi1': 0, 'spi2': 0}, inplace=True)

    model = app.config['model']
    features = app.config['features']
    upcoming['home_win_prob'] = model.predict_proba(upcoming[features])[:, 1]

    # Render the template with the predicted probabilities
    return render_template('scores.html', games=upcoming[['date', 'team1', 'team2', 'home_win_prob']])
