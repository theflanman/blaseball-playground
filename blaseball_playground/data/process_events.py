import json

import numpy as np
import pandas as pd
import tqdm

from blaseball_playground.utils.player_stlats import get_player_stlats, FORBIDDEN_STLATS, RATING_STLATS, STARS


def dict_raw_data(raw_data):
    base_runners = {runner['id']: runner for runner in raw_data['base_runners']}
    game_events = {game_event['id']: game_event for game_event in raw_data['game_events']}
    outcomes = {outcome['id']: outcome for outcome in raw_data['outcomes']}
    return base_runners, game_events, outcomes


def flatten_game_events(game_events, full_hit_data=False):
    """
    Relevant data:
      - batter stlatline
      - pitcher stlatline
      - is called strike
      - is swinging strike
      - is ball
      - is foul ball
      - hit_data
        - either:
          - is_hit
        - or:
          - is fly ball
          - is ground ball
          - is line drive
          - is pop-up
      - is bunt

    :param game_events:
    :return:
    """

    sub_events = []
    misses = 0

    for event_id in tqdm.tqdm(game_events):
        try:
            game_event = game_events[event_id]

            batter = get_player_stlats(game_event['batter_id'], perceived_at=game_event['perceived_at'])
            batter_stats = {f'batter_{stat}': batter[stat] for stat in FORBIDDEN_STLATS + RATING_STLATS}
            pitcher = get_player_stlats(game_event['pitcher_id'], perceived_at=game_event['perceived_at'])
            pitcher_stats = {f'pitcher_{stat}': pitcher[stat] for stat in FORBIDDEN_STLATS + RATING_STLATS}

            pitches = []
            for pitch in game_event['pitches']:
                pitch_data = {
                    'called_strike': pitch == 'C',
                    'swinging_strike': pitch == 'S',
                    'ball': pitch == 'B',
                    'foul': pitch == 'F',
                }
                if full_hit_data:
                    hit_data = {
                        'fly_ball': pitch == 'X' and game_event['batted_ball_type'] == 'FLY',
                        'ground_ball': pitch == 'X' and game_event['batted_ball_type'] == 'GROUNDER',
                        'line_drive': pitch == 'X' and game_event['batted_ball_type'] == 'LINE_DRIVE',
                        'pop_up': pitch == 'X' and game_event['batted_ball_type'] == 'POP_UP',
                    }
                else:
                    hit_data = {'hit': pitch == 'X'}
                pitch_data.update(hit_data)
                pitch_data.update(batter_stats)
                pitch_data.update(pitcher_stats)
                pitches.append(pitch_data)

            sub_events += pitches
        except Exception as e:
            misses += 1

    event_frame = pd.DataFrame(sub_events).astype(np.float64)

    return event_frame


def main():
    files = [
        './blaseball_playground/data/files/events/season_1.json',
        './blaseball_playground/data/files/events/season_2.json',
        './blaseball_playground/data/files/events/season_3.json',
        './blaseball_playground/data/files/events/season_4.json',
        './blaseball_playground/data/files/events/season_5.json',
        './blaseball_playground/data/files/events/season_6.json',
        './blaseball_playground/data/files/events/season_7.json',
    ]

    base_runners = {}
    game_events = {}
    outcomes = {}

    for filename in tqdm.tqdm(files):

        with open(filename, 'r') as f:
            raw_data = json.load(f)

        base_runners_, game_events_, outcomes_ = dict_raw_data(raw_data)
        base_runners.update(base_runners_)
        game_events.update(game_events_)
        outcomes.update(outcomes_)

    flat_events = flatten_game_events(game_events)

    flat_events.to_hdf('./blaseball_playground/data/files/events/events.hdf5', 'events')


if __name__ == '__main__':
    main()
