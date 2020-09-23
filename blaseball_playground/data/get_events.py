import json
import requests
import time

import tqdm

for season in tqdm.tqdm(range(9)):
    season_request = requests.get('https://api.blaseball-reference.com/v1/data/events', params={'season': season})

    with open(f'./blaseball_playground/data/files/events/season_{season}.json', 'w') as f:
        json.dump(json.loads(season_request.content), f, sort_keys=True, indent=4)

    time.sleep(4)
