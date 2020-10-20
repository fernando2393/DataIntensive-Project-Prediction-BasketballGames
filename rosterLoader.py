import glob
import json
import requests
import os
from time import sleep

PATH = "roster/"


def get_rosters(season):
    files = glob.glob(PATH + str(season) + "/" + '**/*.json', recursive=True)
    rosters = {}
    for file in files:
        with open(file) as f:
            data_list = json.load(f)
            aux = rosters.get(data_list['Team_id'])
            if aux is not None:
                aux.append(data_list['Player_id'])
                rosters[data_list['Team_id']] = aux
            else:
                rosters[data_list['Team_id']] = [data_list['Player_id']]

    return rosters


def main():
    session = requests.Session()
    cnt = 0
    for season in range(2002, 2016):
        for team_id in range(1, 31):
            rosters = get_rosters(season)
            values = rosters.get(team_id)
            if values is None:
                continue
            values.sort()
            url = "https://www.balldontlie.io/api/v1/season_averages?season="+str(season)
            for idx in values:
                url += "&player_ids[]=" + str(idx)
            data = session.get(url)
            if data.status_code != 200:
                print(data.status_code)
            json_file = data.json()
            try:
                os.stat("season_averages/" + str(season) + "/")
            except:
                try:
                    os.mkdir("season_averages/" + str(season) + "/")
                except:
                    os.mkdir("season_averages/")
                    os.mkdir("season_averages/" + str(season) + "/")
            with open("season_averages/" + str(season) + "/" + str(team_id) + ".json", 'w') as f:
                json.dump(json_file, f)
                print("Season " + str(season) + "\tTeam " + str(team_id) + " saved")
                cnt += 1
            if cnt % 60 == 0 and cnt != 0:
                sleep(60)


if __name__ == "__main__":
    main()
