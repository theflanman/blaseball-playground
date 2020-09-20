import os
import sys

import json
import pandas as pd
import numpy as np
import tqdm


class Game:

    def __init__(self, game_df, game_id):
        self._game_df = game_df
        self.game_id = game_id

        self._make_first_last_inning_event()
        self._make_scores()
        self._make_teams()

        pass

    def __repr__(self):
        return f'<Game {self.home_score}-{self.away_score} {self.game_id}>'

    def _make_first_last_inning_event(self):
        if len(np.unique(self._game_df['perceived_at'])) == 1:
            inning_value = (self._game_df['inning'].array.astype(float) + 1 - 0.5*self._game_df['top_of_inning']).array

            last_inning = self._game_df.iloc[inning_value == inning_value.max(), :]
            last_inning_score_total = np.array(last_inning['away_score'] + last_inning['home_score'], dtype=np.float)
            self.last_inning_event = last_inning.iloc[last_inning_score_total.argmax(), :]

            first_inning = self._game_df.iloc[inning_value == inning_value.min(), :]
            first_inning_score_total = np.array(first_inning['away_score'] + first_inning['home_score'], dtype=np.float)
            self.last_inning_event = last_inning.iloc[first_inning_score_total.argmin(), :]
        else:
            self.first_inning_event = self._game_df.iloc[0, :]
            self.last_inning_event = self._game_df.iloc[-1, :]

    def _make_scores(self):
        self.away_score = self.last_inning_event['away_score']
        self.home_score = self.last_inning_event['home_score']

    def _make_teams(self):
        if self.last_inning_event['top_of_inning']:
            self.away_team_id = self.last_inning_event['pitcher_team_id']
            self.home_team_id = self.last_inning_event['batter_team_id']
        else:
            self.home_team_id = self.last_inning_event['pitcher_team_id']
            self.away_team_id = self.last_inning_event['batter_team_id']
        pass


def load_games(file_name):
    with open(f'./events/{file_name}', 'r') as f:
        blaseball_ref = json.load(f)

    if len(blaseball_ref['game_events']) == 0:
        return []

    outcomes = pd.DataFrame(blaseball_ref['outcomes'])
    game_events = pd.DataFrame(blaseball_ref['game_events'])
    game_events['perceived_at'] = pd.to_datetime(game_events['perceived_at'])
    game_events.sort_values('perceived_at')
    base_runners = pd.DataFrame(blaseball_ref['base_runners'])

    game_ids = list(set(game_events.loc[:, 'game_id'].to_list()))

    games = [Game(game_events[game_events['game_id'] == game_id], game_id) for game_id in game_ids]
    game = games[0]

    return games


def main():
    # load games from all files in ./events/
    games = []

    for file in tqdm.tqdm(os.listdir('./events/')):
        games += load_games(file)

    return games


if __name__ == '__main__':
    print(len(main()))
