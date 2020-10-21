import glob
import pandas as pd
import numpy as np
from Features import Features as ft

SEASON_AVG = "season_averages/"


def dataLoader(local, visitor, season, features, averaged=True):
    local_team = np.zeros((12, 22))
    visitor_team = np.ones((12, 22))
    id_1 = local
    id_2 = visitor
    if averaged:
        id_1 = str(local) + "-36"
        id_2 = str(visitor) + "-36"
    files = glob.glob(SEASON_AVG + str(season) + "/" + id_1 + '/*.json', recursive=False)  # Local team
    for file in files:
        json_file = pd.read_json(file)
        for idx, row in json_file.iterrows():
            local_team[idx] = np.array(list(row['data'].values()))

    files = glob.glob(SEASON_AVG + str(season) + "/" + id_2 + '/*.json', recursive=False)  # Visitor team
    for file in files:
        json_file = pd.read_json(file)
        for idx, row in json_file.iterrows():
            visitor_team[idx] = np.array(list(row['data'].values()))

    if len(features) > 0:
        local_team = local_team[:, features]
        visitor_team = visitor_team[:, features]
    local_team = np.append(local_team, np.zeros((12, 1)), axis=1)
    visitor_team = np.append(visitor_team, np.ones((12, 1)), axis=1)

    return local_team, visitor_team


def main():
    # local_team, visitor_team = dataLoader(22, 14, 2010, [ft.ast.value], True)

    for season in range(2002, 2015):
        json_file = pd.read_json("data/game_data.json")
        json_file = json_file['data']
        json_file = json_file[json_file['season'] == season]
        pass

if __name__ == "__main__":
    main()
