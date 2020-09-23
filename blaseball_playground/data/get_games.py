import itertools
import json
import requests
import time
import os

import numpy as np
import pandas as pd
import tqdm


def season_day_to_datetime(season, day):
    season_start_dates = {
        0:  np.datetime64('2020-07-20'),
        1:  np.datetime64('2020-07-27'),
        2:  np.datetime64('2020-08-03'),
        3:  np.datetime64('2020-08-24'),
        4:  np.datetime64('2020-08-31'),
        5:  np.datetime64('2020-09-07'),
        6:  np.datetime64('2020-09-14'),
        7:  np.datetime64('2020-09-21'),
        8:  np.datetime64('2020-09-28'),
        9:  np.datetime64('2020-10-05'),
        10: np.datetime64('2020-10-12'),
    }

    dt = season_start_dates[season] + np.timedelta64(5, 'h')  # adjusting for est

    if day <= 100:  # on season games
        dt += np.timedelta64(day, 'h')
    else:
        dt += np.timedelta64(6, 'D')  # move time to saturday
        dt += np.timedelta64(day - 100, 'h')

    return dt


class Game:
    def __init__(self,
                 game,
                 game_statsheet,
                 away_team_statsheet,
                 home_team_statsheet,
                 away_team_player_statsheets,
                 home_team_player_statsheets,
                 away_team_offense,
                 away_team_defense,
                 home_team_offense,
                 home_team_defense,
                 relevant_stats,
                 ):
        self.id = game['id']
        self.rules = game['rules']
        self.statsheet = game['statsheet']
        self.awayTeam = game['awayTeam']
        self.awayTeamName = game['awayTeamName']
        self.awayTeamNickname = game['awayTeamNickname']
        self.awayOdds = game['awayOdds']
        self.awayScore = game['awayScore']
        self.homeTeam = game['homeTeam']
        self.homeTeamName = game['homeTeamName']
        self.homeTeamNickname = game['homeTeamNickname']
        self.homeOdds = game['homeOdds']
        self.homeScore = game['homeScore']
        self.season = game['season']
        self.day = game['day']
        self.game_datetime = season_day_to_datetime(self.season, self.day)
        self.relevant_stats = relevant_stats

        self.valid = True

        self.awayTeamOffense = self.calc_avg_stats(away_team_offense)
        self.awayTeamDefense = self.calc_avg_stats(away_team_defense)
        self.homeTeamOffense = self.calc_avg_stats(home_team_offense)
        self.homeTeamDefense = self.calc_avg_stats(home_team_defense)

    def __repr__(self):
        return f'<Game {self.homeTeamNickname} vs {self.awayTeamNickname}: {self.homeScore}-{self.awayScore}>'

    def to_dict(self):

        out = {'away_score': self.awayScore, 'home_score': self.homeScore}

        for i, player in enumerate(self.awayTeamOffense):
            out.update({f'away_offense_{i}_{stat}': player[stat] for stat in self.relevant_stats})
        for i, player in enumerate(self.awayTeamDefense):
            out.update({f'away_defense_{i}_{stat}': player[stat] for stat in self.relevant_stats})
        for i, player in enumerate(self.homeTeamOffense):
            out.update({f'home_offense_{i}_{stat}': player[stat] for stat in self.relevant_stats})
        for i, player in enumerate(self.homeTeamDefense):
            out.update({f'home_defense_{i}_{stat}': player[stat] for stat in self.relevant_stats})

        return out

    def calc_player_stats(self, player_statsheet, player_fk):
        player_fk = pd.DataFrame(player_fk, columns=['valid_from', 'valid_until'] + self.relevant_stats)[
            ['valid_from', 'valid_until'] + self.relevant_stats]
        player_fk['valid_from'] = pd.to_datetime(player_fk['valid_from'])
        player_fk['valid_until'] = pd.to_datetime(player_fk['valid_until'])
        for stat in self.relevant_stats:
            player_fk[stat] = player_fk[stat].astype(np.float64)

        return player_fk.mean()

    def calc_avg_stats(self, players):
        if len(players) == 0:
            self.valid = False
            return {stat: 0 for stat in self.relevant_stats}
        player_fks = []
        for statsheet, forbidden in players:
            if len(forbidden) == 0:
                self.valid = False
                return {stat: 0 for stat in self.relevant_stats}
            forbidden.sort(key=lambda x: np.datetime64(x['valid_from']))
            for fk in forbidden:
                if fk['valid_until'] is None:
                    fk['valid_until'] = np.datetime64('now')
            valid_from = np.array([fk['valid_from'] for fk in forbidden], dtype=np.datetime64)
            # valid_until = np.array([fk['valid_until'] for fk in forbidden], dtype=np.datetime64)
            if self.game_datetime < valid_from[0]:
                fk = forbidden[0]
                self.valid = False
            for fk in forbidden:
                if self.game_datetime >= np.datetime64(fk['valid_from']):
                    break
            player_fks.append({stat:fk[stat] for stat in self.relevant_stats})

        out = player_fks

        # for fk in player_fks:
        #     for stat in self.relevant_stats:
        #         out[stat] += float(fk[stat])/len(player_fks)

        return out


STAR_STATS = [
    'batting_rating',
    'baserunning_rating',
    'defense_rating',
    'pitching_rating',
]


FORBIDDEN_STATS = [
    'base_thirst',
    'continuation',
    'ground_friction',
    'indulgence',
    'laserlikeness',

    'anticapitalism',
    'chasiness',
    'omniscience',
    'tenaciousness',
    'watchfulness',

    'buoyancy',
    'divinity',
    'martyrdom',
    'moxie',
    'musclitude',
    'patheticism',
    'thwackability',
    'tragicness',

    'coldness',
    'overpowerment',
    'ruthlessness',
    'shakespearianism',
    'suppression',
    'unthwackability',
]


def get_season_info():

    filename = './blaseball_playground/data/files/seasons.json'
    if not os.path.exists(filename):
        seasons = []
        season = 0
        while True:
            season_req = requests.get('https://www.blaseball.com/database/season', params={'number': season})
            if season_req.content == b'':
                break
            season_req = season_req.json()

            postseason_req = requests.get('https://www.blaseball.com/database/playoffs', params={'number': season}).json()

            seasons.append((season_req, postseason_req, postseason_req['playoffDay'] + 1))
            season += 1

        with open(filename, 'w') as f:
            json.dump(seasons, f)
            
    else:
        with open(filename, 'r') as f:
            seasons = json.load(f)

    return seasons


def get_games(n_seasons, postseason_lens):

    filename = './blaseball_playground/data/files/games.json'
    if not os.path.exists(filename):
    
        # There's a better way to do this so that tqdm cooperates but whatever
        day_tuples = []
        for season in range(n_seasons):
            for day in range(100 + postseason_lens[season]):
                day_tuples.append((season, day))
    
        games = []
        for season, day in tqdm.tqdm(day_tuples):
            req = requests.get('https://www.blaseball.com/database/games', params={'day': day, 'season': season})
            games += req.json()
            time.sleep(0.1)
    
        with open(filename, 'w') as f:
            json.dump(games, f, indent=4, sort_keys=True)

    else:
        with open(filename, 'r') as f:
            games = json.load(f)

    return games


def get_game_statsheets(games):

    filename = './blaseball_playground/data/files/game_statsheets.json'
    if not os.path.exists(filename):

        game_statsheet_ids = [game['statsheet'] for game in games]
        game_statsheets = []

        bundle_size = 10
        for i in tqdm.tqdm(range(0, len(game_statsheet_ids), bundle_size)):
            bundle = game_statsheet_ids[i:i+bundle_size]
            req = requests.get('https://www.blaseball.com/database/gameStatsheets', params={'ids': ','.join(bundle)})
            game_statsheets += req.json()
            time.sleep(0.1)

        with open(filename, 'w') as f:
            json.dump(game_statsheets, f)
    else:
        with open(filename, 'r') as f:
            game_statsheets = json.load(f)

    return game_statsheets


def get_team_statsheets(team_statsheets_ids):

    filename = './blaseball_playground/data/files/team_statsheets.json'
    if not os.path.exists(filename):

        team_statsheets = []

        bundle_size = 10
        for i in tqdm.tqdm(range(0, len(team_statsheets_ids), bundle_size)):
            bundle = team_statsheets_ids[i:i+bundle_size]
            req = requests.get('https://www.blaseball.com/database/teamStatsheets', params={'ids': ','.join(bundle)})
            team_statsheets += req.json()
            time.sleep(0.1)

        with open(filename, 'w') as f:
            json.dump(team_statsheets, f)

    else:
        with open(filename, 'r') as f:
            team_statsheets = json.load(f)

    return team_statsheets


def get_player_statsheets(player_statsheet_ids):

    filename = './blaseball_playground/data/files/player_statsheets.json'
    if not os.path.exists(filename):

        player_statsheets = []

        bundle_size = 100
        for i in tqdm.tqdm(range(0, len(player_statsheet_ids), bundle_size)):
            bundle = player_statsheet_ids[i:i+bundle_size]
            req = requests.get('https://www.blaseball.com/database/playerSeasonStats', params={'ids': ','.join(bundle)})
            player_statsheets += req.json()
            time.sleep(0.1)

        with open(filename, 'w') as f:
            json.dump(player_statsheets, f)
    else:
        with open(filename, 'r') as f:
            player_statsheets = json.load(f)

    return player_statsheets


def get_players(player_ids, ):

    filename = './blaseball_playground/data/files/players.json'
    if not os.path.exists(filename):

        players = []
        for player_id in tqdm.tqdm(player_ids):
            req = requests.get('https://api.blaseball-reference.com/v1/playerInfo', params={'playerId': player_id, 'all': 'true'})
            players.append(req.json())
            time.sleep(0.1)

        with open(filename, 'w') as f:
            json.dump(players, f)
    else:
        with open(filename, 'r') as f:
            players = json.load(f)
    return players


def make_games(games, game_statsheets, team_statsheets, player_statsheets, player_dict, relevant_stats):
    game_statsheets = {game_statsheet['id']: game_statsheet for game_statsheet in game_statsheets}
    team_statsheets = {team_statsheet['id']: team_statsheet for team_statsheet in team_statsheets}
    player_statsheets = {player_statsheet['id']: player_statsheet for player_statsheet in player_statsheets}

    game_objs = []

    for game in tqdm.tqdm(games):

        game_statsheet = game_statsheets[game['statsheet']]

        away_team_statsheet = team_statsheets[game_statsheet['awayTeamStats']]
        home_team_statsheet = team_statsheets[game_statsheet['homeTeamStats']]

        away_team_player_statsheets = [player_statsheets[player_stat_id] for player_stat_id in away_team_statsheet['playerStats']]
        home_team_player_statsheets = [player_statsheets[player_stat_id] for player_stat_id in home_team_statsheet['playerStats']]

        none_players = sum([player_statsheet['playerId'] is None for player_statsheet in away_team_player_statsheets + home_team_player_statsheets])

        if none_players > 0:
            continue

        away_team_offense = [(atps, player_dict[atps['playerId']]) for atps in away_team_player_statsheets if atps['atBats'] > 0]
        away_team_defense = [(atps, player_dict[atps['playerId']]) for atps in away_team_player_statsheets if atps['atBats'] == 0]
        home_team_offense = [(htps, player_dict[htps['playerId']]) for htps in away_team_player_statsheets if htps['atBats'] > 0]
        home_team_defense = [(htps, player_dict[htps['playerId']]) for htps in away_team_player_statsheets if htps['atBats'] == 0]

        game_objs.append(Game(game,
                              game_statsheet,
                              away_team_statsheet,
                              home_team_statsheet,
                              away_team_player_statsheets,
                              home_team_player_statsheets,
                              away_team_offense,
                              away_team_defense,
                              home_team_offense,
                              home_team_defense,
                              relevant_stats,
                              ))

    return pd.DataFrame([game_obj.to_dict() for game_obj in game_objs if game_obj.valid])


def main():

    # Collect a ton of data.  Goal is to have a list of all games, with all players present in that game.  Then, compile
    # a list of all players and their positions on each team during those games
    season_info = get_season_info()
    n_seasons = len(season_info)
    postseason_lens = [s[2] for s in season_info]

    games = get_games(n_seasons, postseason_lens)

    game_statsheets = get_game_statsheets(games)
    team_statsheet_ids = [s['awayTeamStats'] for s in game_statsheets] + [s['homeTeamStats'] for s in game_statsheets]

    team_statsheets = get_team_statsheets(team_statsheet_ids)
    player_statsheet_ids = []
    for team_statsheet in team_statsheets:
        player_statsheet_ids += team_statsheet['playerStats']

    player_statsheets = get_player_statsheets(player_statsheet_ids)

    player_ids = []
    for player_statsheet in tqdm.tqdm(player_statsheets):
        if player_statsheet['playerId'] is None:
            continue
        player_ids.append(player_statsheet['playerId'])
    player_ids = list(set(player_ids))

    players = get_players(player_ids)
    player_dict = {pid: [p for p in players if p['player_id'] == pid] for pid in tqdm.tqdm(player_ids)}

    star_game_stats = make_games(games, game_statsheets, team_statsheets, player_statsheets, player_dict, STAR_STATS)
    star_game_stats.to_csv('./blaseball_playground/data/files/star_game_stats.csv')

    forbidden_game_stats = make_games(games, game_statsheets, team_statsheets, player_statsheets, player_dict, FORBIDDEN_STATS)
    forbidden_game_stats.to_csv('./blaseball_playground/data/files/forbidden_game_stats.csv')


if __name__ == '__main__':
    main()
