import json
import requests
import time

import tqdm


def get_rosters():
    pass


def get_deceased():
    pass


def main():
    deceased = get_deceased()

    rosters = get_rosters()

player_id = '3d3be7b8-1cbf-450d-8503-fce0daf46cbf'

req = requests.get('https://api.blaseball-reference.com/v1/playerInfo', params={'playerId': player_id})

pass
