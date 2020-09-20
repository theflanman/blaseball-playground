import json
import requests
import time

import tqdm

for season in tqdm.tqdm(range(7)):
    season_request = requests.get('https://api.blaseball-reference.com/v1/data/events', params={'season': season})

    with open(f'./events/season_{season}.json', 'w') as f:
        json.dump(json.loads(season_request.content), f)

    time.sleep(4)
