import glob
import json
import requests
import pandas as pd

PATH = "test/"


def get_rosters(season):
    files = glob.glob(PATH+'**/*.json', recursive=True)
    rosters = {}
    for file in files:
        with open(file) as f:
            if file.split("/")[1] == str(season):
                data_list = json.load(f)
                aux = rosters.get(data_list['team_id'])
                if aux is not None:
                    aux.append(data_list['player_id'])
                    rosters[data_list['team_id']] = aux
                else:
                    rosters[data_list['team_id']] = [data_list['player_id']]

    return rosters


def main():
    session = requests.Session()
    season = 288
    rosters = get_rosters(season)
    values = rosters[23]
    values.sort()
    uri = "https://www.balldontlie.io/api/v1/season_averages?season=2018"
    for idx in values:
        uri += "&player_ids[]="+str(idx)
    print(uri)
    data = session.get(uri)
    print(data.status_code)
    json_file = data.json()
    print(json_file)


if __name__ == "__main__":
    main()
