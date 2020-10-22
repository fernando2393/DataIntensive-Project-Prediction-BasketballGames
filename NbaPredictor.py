import glob
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from Features import Stats as st
from Features import Games as gm

SEASON_AVG = "season_averages/"
SEASON_GAMES = "season_games/"


def teamLoader(local, visitor, season, features, averaged=True):
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


def gameLoader(season):
    file = glob.glob(SEASON_GAMES + str(season) + '/*.json', recursive=False)  # Season games
    json_file = pd.read_json(file[0])
    games = np.zeros((json_file.shape[0], len(json_file.iloc[0][0]) - 1))  # Remove the date
    for idx, row in json_file.iterrows():
        games[idx] = np.array(list(row['data'].values())[1:])

    return games


def dataLoader(season_start, season_end, features):
    x = list()
    y = list()
    seasons = list()
    for season in range(season_start, season_end + 1):
        games = gameLoader(season)
        seasons.append(season)
        for game in tqdm(games):
            local_team, visitor_team = teamLoader(int(game[gm.home_team_id.value]),
                                                  int(game[gm.visitor_team_id.value]),
                                                  season,
                                                  features,
                                                  True)
            x.append(np.vstack((local_team, visitor_team)))
            y.append(np.argmax([game[gm.home_team_score.value], game[gm.visitor_team_score.value]]))
    print("Seasons obtained " + str(seasons))

    return x, y


def main():
    # Training data
    features = [
                st.ast.value,
                st.blk.value,
                st.dreb.value,
                # st.fg3_pct.value,
                st.fg3a.value,
                st.fg3m.value,
                # st.fg_pct.value,
                st.fga.value,
                st.fgm.value,
                # st.ft_pct.value,
                st.fta.value,
                st.ftm.value,
                st.games_played.value,
                # st.seconds.value,
                st.oreb.value,
                st.pf.value,
                # st.player_id.value,
                st.pts.value,
                # st.reb.value,
                # st.season.value,
                st.stl.value,
                st.turnover.value
    ]
    print("Loading training data...")
    if os.path.exists("train.npz"):
        npz_file = np.load("train.npz")
        x_train = npz_file['x']
        y_train = npz_file['y']
    else:
        x_train, y_train = dataLoader(2002, 2014, features)  # Both initial and end are included
        np.savez("train.npz", x=x_train, y=y_train)
    training_samples = len(x_train)
    # Testing data
    print("Loading testing data...")
    if os.path.exists("test.npz"):
        npz_file = np.load("test.npz")
        x_test = npz_file['x']
        y_test = npz_file['y']
    else:
        x_test, y_test = dataLoader(2015, 2015, features)  # Both initial and end are included
        np.savez("test.npz", x=x_test, y=y_test)
    testing_samples = len(x_test)
    # Scale training data
    x_to_scale = np.vstack(x_train)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_to_scale[:, :len(features)])
    x_train = np.hstack((x_train, x_to_scale[:, -1].reshape((-1, 1))))
    x_train = [x_train[i * 24:24 * (i + 1), :] for i in range(training_samples)]
    x_train = [x_train[i].flatten() for i in range(len(x_train))]
    # Scale testing data
    x_to_scale = np.vstack(x_test)
    x_test = scaler.transform(x_to_scale[:, :len(features)])
    x_test = np.hstack((x_test, x_to_scale[:, -1].reshape((-1, 1))))
    x_test = [x_test[i * 24:24 * (i + 1), :] for i in range(testing_samples)]
    x_test = [x_test[i].flatten() for i in range(len(x_test))]

    clf = MLPClassifier(hidden_layer_sizes=(40, 28, 10), early_stopping=True, max_iter=250,
                        solver="adam",
                        activation="logistic"
                        ).fit(x_train, y_train)
    print("MLP: Accuracy on testing: " + str(clf.score(x_test, y_test)))

    # clf = SVC()
    # clf.fit(x_train, y_train)
    # print("SVM\tAccuracy on testing: " + str(clf.score(x_test, y_test)))

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(x_train, y_train)
    print("Random Forest: Accuracy on testing: " + str(clf.score(x_test, y_test)))


if __name__ == "__main__":
    main()
