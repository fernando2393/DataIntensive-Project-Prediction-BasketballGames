import requests
import json
import time

MAX_CALLS = 60

parameters = {
    "per_page": 100,
    "page": 0,
}
session = requests.Session()
data = session.get("https://www.balldontlie.io/api/v1/games", params=parameters)
print(data.status_code)

json_file = data.json()
print(json_file['meta'])

data_list = []
for i in range(1, 487):
    if i % (MAX_CALLS + 1) != 0:
        print("Call: " + str(i))
        parameters = {
            "per_page": 100,
            "page": i,
        }
        aux = session.get("https://www.balldontlie.io/api/v1/games", params=parameters)
        assert aux.status_code == 200, "Status code is " + str(aux.status_code)
        data_list.append(aux.json())
    else:
        time.sleep(60)  # Wait for one minute and do other 60 API requests

with open('game_data.json', 'w') as f:
    json.dump(data_list, f)
