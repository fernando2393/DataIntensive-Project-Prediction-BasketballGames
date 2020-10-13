import requests
import json
import time

MAX_CALLS = 60

parameters = {
    "per_page": 100,
    "page": 0,
}
session = requests.Session()
data = session.get("https://www.balldontlie.io/api/v1/stats", params=parameters)
print(data.status_code)

json_file = data.json()
print(json_file['meta'])

with open('stats_data.json') as f:
    data_list = json.load(f)

for i in range(704, json_file['meta']['total_pages']):
    parameters = {
        "per_page": 100,
        "page": i,
    }
    if i == 0 or (i + 1) % MAX_CALLS != 0:
        print("Request: " + str(i))
        aux = session.get("https://www.balldontlie.io/api/v1/stats", params=parameters)
        assert aux.status_code == 200, "Status code is " + str(aux.status_code)
        data_list.append(aux.json())
    else:
        time.sleep(60)  # Wait for one minute and do other 60 API requests
        print("Request: " + str(i))
        aux = session.get("https://www.balldontlie.io/api/v1/stats", params=parameters)
        assert aux.status_code == 200, "Status code is " + str(aux.status_code)
        data_list.append(aux.json())

    if i % 100 == 0:
        with open('stats_data.json', 'w') as f:
            json.dump(data_list, f)
        print("Saving checkpoint...")

with open('stats_data.json', 'w') as f:
    json.dump(data_list, f)
