import json

import numpy as np
import pandas as pd


PLAYER_STLAT_DATA = {}

FLOAT_STATS = [
    'anticapitalism',
    'base_thirst',
    'buoyancy',
    'chasiness',
    'coldness',
    'continuation',
    'divinity',
    'ground_friction',
    'indulgence',
    'laserlikeness',
    'martyrdom',
    'moxie',
    'musclitude',
    'omniscience',
    'overpowerment',
    'patheticism',
    'ruthlessness',
    'shakespearianism',
    'suppression',
    'tenaciousness',
    'thwackability',
    'tragicness',
    'unthwackability',
    'watchfulness',
    'pressurization',
    'cinnamon',
    'batting_rating',
    'baserunning_rating',
    'defense_rating',
    'pitching_rating',
    'batting_stars',
    'baserunning_stars',
    'defense_stars',
    'pitching_stars',
]

FORBIDDEN_STLATS = [
    'anticapitalism',
    'base_thirst',
    'buoyancy',
    'chasiness',
    'coldness',
    'continuation',
    'divinity',
    'ground_friction',
    'indulgence',
    'laserlikeness',
    'martyrdom',
    'moxie',
    'musclitude',
    'omniscience',
    'overpowerment',
    'patheticism',
    'ruthlessness',
    'shakespearianism',
    'suppression',
    'tenaciousness',
    'thwackability',
    'tragicness',
    'unthwackability',
    'watchfulness',
    'pressurization',
    'cinnamon',
]

RATING_STLATS = [
    'batting_rating',
    'baserunning_rating',
    'defense_rating',
    'pitching_rating',
]

STARS = [
    'batting_stars',
    'baserunning_stars',
    'defense_stars',
    'pitching_stars',
]


def create_stlatlines():
    with open('./blaseball_playground/data/files/players.json', 'r') as f:
        players = json.load(f)

    for player in players:

        # datatype conversions
        if player['valid_until'] is None:
            player['valid_until'] = np.datetime64('2030')
        else:
            player['valid_until'] = np.datetime64(player['valid_until'])
        player['valid_from'] = np.datetime64(player['valid_from'])

        for k in FLOAT_STATS:
            player[k] = float(player[k])


        if player['player_id'] in PLAYER_STLAT_DATA:
            PLAYER_STLAT_DATA[player['player_id']].append(player)
        else:
            PLAYER_STLAT_DATA[player['player_id']] = [player]

    for player_id in PLAYER_STLAT_DATA:
        PLAYER_STLAT_DATA[player_id] = pd.DataFrame(PLAYER_STLAT_DATA[player_id]).sort_values('valid_from')


def get_player_stlats(player_id, perceived_at=None, season=None, day=None):

    if perceived_at is None:
        perceived_at = np.datetime64('now')
    else:
        perceived_at = np.datetime64(perceived_at)

    if len(PLAYER_STLAT_DATA) == 0:
        create_stlatlines()

    for i, row in PLAYER_STLAT_DATA[player_id].iterrows():
        if row['valid_from'] >= perceived_at:
            return row
        elif row['valid_until'] <= perceived_at:
            return row
        elif row['valid_until'] > perceived_at > row['valid_from']:
            return row

    raise KeyError('I dunno, couldn"t find it')