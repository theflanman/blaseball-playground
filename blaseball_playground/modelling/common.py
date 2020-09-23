import numpy as np
import pandas as pd


def get_data(name):
    if name == 'star':
        data = pd.read_csv('./blaseball_playground/data/files/star_game_stats.csv')
    else:
        data = pd.read_csv('./blaseball_playground/data/files/forbidden_game_stats.csv')
    data = data[[c for c in data.columns if 'Unnamed' not in c]]

    # eliminate nan-heavy columns
    max_nan = len(data)//10
    data = data[[c for c in data.columns if np.isnan(data[c]).sum() <= max_nan]]
    data = data.dropna()


    X = data[[c for c in data.columns if c not in ['away_score', 'home_score']]].to_numpy()
    y = data[['away_score', 'home_score']].to_numpy()

    return X, y.astype(np.float64)